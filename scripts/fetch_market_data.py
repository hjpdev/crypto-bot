#!/usr/bin/env python3
"""
Market data fetching script.

This script demonstrates how to use the exchange service to fetch
and display market data from cryptocurrency exchanges.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from tabulate import tabulate

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.exchange_service import ExchangeService
from app.services.data_normalization import normalize_ohlcv, normalize_ticker, normalize_order_book, normalize_trades, normalize_funding_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("fetch_market_data")

# @TODO: Load exchange API keys from config file
# def load_exchange_api_keys() -> Dict[str, Dict[str, str]]:
#     """Load exchange API keys from environment variables."""
#     exchanges = {}

#     # Binance
#     binance_api_key = os.environ.get("BINANCE_API_KEY")
#     binance_secret = os.environ.get("BINANCE_SECRET")
#     if binance_api_key and binance_secret:
#         exchanges["binance"] = {
#             "api_key": binance_api_key,
#             "secret": binance_secret
#         }

#     # Coinbase Pro
#     coinbase_api_key = os.environ.get("COINBASE_API_KEY")
#     coinbase_secret = os.environ.get("COINBASE_SECRET")
#     coinbase_password = os.environ.get("COINBASE_PASSPHRASE")
#     if coinbase_api_key and coinbase_secret and coinbase_password:
#         exchanges["coinbasepro"] = {
#             "api_key": coinbase_api_key,
#             "secret": coinbase_secret,
#             "password": coinbase_password
#         }

#     # Kraken
#     kraken_api_key = os.environ.get("KRAKEN_API_KEY")
#     kraken_secret = os.environ.get("KRAKEN_SECRET")
#     if kraken_api_key and kraken_secret:
#         exchanges["kraken"] = {
#             "api_key": kraken_api_key,
#             "secret": kraken_secret
#         }

#     return exchanges


def display_table(data: List[Dict[str, Any]], title: str) -> None:
    """Display data in a nicely formatted table."""
    if not data:
        logger.info(f"{title}: No data")
        return

    try:
        df = pd.DataFrame(data)

        if len(df.columns) > 10:
            most_important = ["symbol", "exchange", "timestamp", "datetime", "open", "high", "low",
                             "close", "volume", "price", "amount", "side"]
            columns_to_display = [col for col in most_important if col in df.columns]
            remaining_cols = [col for col in df.columns if col not in columns_to_display]
            columns_to_display.extend(remaining_cols[:10 - len(columns_to_display)])
            df = df[columns_to_display]

        logger.info(f"\n{title}:\n")
        logger.info(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

    except Exception as e:
        logger.error(f"Error displaying table: {e}")
        logger.info(f"{title}: {data}")


def fetch_ohlcv_data(exchange_service: ExchangeService, args: argparse.Namespace) -> None:
    """Fetch and display OHLCV data."""
    try:
        if args.days:
            start_time = int((datetime.now() - timedelta(days=args.days)).timestamp() * 1000)
        else:
            start_time = None

        logger.info(f"Fetching OHLCV data for {args.symbol} on timeframe {args.timeframe}...")

        ohlcv_data = exchange_service.fetch_historical_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_time=start_time,
            limit=args.limit
        )

        normalized_data = normalize_ohlcv(
            ohlcv_data,
            exchange_service.exchange_id,
            args.symbol
        )

        display_table(normalized_data[-20:], f"Last 20 OHLCV candles for {args.symbol}")

        if normalized_data:
            df = pd.DataFrame(normalized_data)
            logger.info("\nBasic statistics:")
            logger.info(f"Total candles: {len(df)}")
            logger.info(f"Date range: {df['datetime'].min()} - {df['datetime'].max()}")
            logger.info(f"Price range: {df['low'].min()} - {df['high'].max()}")
            logger.info(f"Average volume: {df['volume'].mean():.2f}")

    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")


def fetch_order_book_data(exchange_service: ExchangeService, args: argparse.Namespace) -> None:
    """Fetch and display order book data."""
    try:
        logger.info(f"Fetching order book for {args.symbol}...")

        order_book = exchange_service.fetch_order_book_snapshot(
            symbol=args.symbol,
            depth=args.depth
        )

        normalized_book = normalize_order_book(
            order_book,
            exchange_service.exchange_id,
            args.symbol,
            args.depth
        )

        if normalized_book:
            logger.info(f"\nOrder book for {args.symbol}:")
            logger.info(f"Timestamp: {normalized_book['datetime']}")

            bids_df = pd.DataFrame(normalized_book['bids'])
            asks_df = pd.DataFrame(normalized_book['asks'])

            if not bids_df.empty and not asks_df.empty:
                display_levels = min(5, len(bids_df), len(asks_df))

                logger.info("\nTop bid levels:")
                logger.info(tabulate(bids_df[:display_levels], headers="keys", tablefmt="psql", showindex=False))

                logger.info("\nTop ask levels:")
                logger.info(tabulate(asks_df[:display_levels], headers="keys", tablefmt="psql", showindex=False))

                top_bid = bids_df.iloc[0]['price'] if not bids_df.empty else None
                top_ask = asks_df.iloc[0]['price'] if not asks_df.empty else None

                if top_bid and top_ask:
                    spread = top_ask - top_bid
                    spread_pct = (spread / top_bid) * 100
                    logger.info(f"\nBest bid: {top_bid}, Best ask: {top_ask}")
                    logger.info(f"Spread: {spread:.8f} ({spread_pct:.4f}%)")
            else:
                logger.warning(f"Order book contains no bids or asks for {args.symbol}")

    except Exception as e:
        logger.exception(f"Error fetching order book: {e}")


def fetch_ticker_data(exchange_service: ExchangeService, args: argparse.Namespace) -> None:
    """Fetch and display ticker data."""
    try:
        if "," in args.symbol:
            symbols = [s.strip() for s in args.symbol.split(",")]
            logger.info(f"Fetching ticker data for {len(symbols)} symbols...")

            tickers = exchange_service.get_ticker_batch(symbols=symbols)

            all_normalized = []
            for symbol, ticker in tickers.items():
                if isinstance(ticker, dict) and 'error' not in ticker:
                    try:
                        normalized = normalize_ticker(ticker, exchange_service.exchange_id, symbol)
                        all_normalized.append(normalized)
                    except Exception as e:
                        logger.error(f"Error normalizing ticker for {symbol}: {str(e)}")
                        if logger.isEnabledFor(logging.DEBUG):
                            import traceback
                            logger.debug(traceback.format_exc())
                else:
                    logger.error(f"Error fetching ticker for {symbol}: {ticker}")

            if all_normalized:
                display_table(all_normalized, "Ticker Data")
            else:
                logger.warning("No valid ticker data found for any of the requested symbols")

        else:
            logger.info(f"Fetching ticker data for {args.symbol}...")

            ticker = exchange_service.get_ticker(symbol=args.symbol)

            if not ticker:
                logger.warning(f"No ticker data returned for {args.symbol}")
                return

            try:
                normalized = normalize_ticker(ticker, exchange_service.exchange_id, args.symbol)
                display_table([normalized], f"Ticker for {args.symbol}")

                # Display extra information about the ticker
                if normalized:
                    logger.info(f"\nLast price: {normalized['last']}")
                    if normalized['bid'] and normalized['ask']:
                        spread = normalized['ask'] - normalized['bid']
                        spread_pct = (spread / normalized['bid']) * 100 if normalized['bid'] else 0
                        logger.info(f"Bid/Ask spread: {spread:.8f} ({spread_pct:.4f}%)")
                    logger.info(f"24h change: {normalized['change']:.8f} ({normalized['percentage']:.2f}%)")
                    logger.info(f"24h volume: {normalized['volume']:.8f}")
            except Exception as e:
                logger.error(f"Error processing ticker for {args.symbol}: {str(e)}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error fetching ticker data: {e}")


def fetch_trade_data(exchange_service: ExchangeService, args: argparse.Namespace) -> None:
    """
    Fetch and display recent trades for a symbol.

    Args:
        exchange_service: The exchange service to use
        args: Command line arguments with symbol and limit
    """
    try:
        logger.info(f"Fetching recent trades for {args.symbol}...")

        trades = exchange_service.fetch_recent_trades(
            symbol=args.symbol,
            limit=args.limit
        )

        if not trades:
            logger.warning(f"No trades returned for {args.symbol}")
            return

        # Log the count of trades received
        logger.info(f"Received {len(trades)} trades")

        try:
            normalized_trades = normalize_trades(
                trades,
                exchange_service.exchange_id,
                args.symbol
            )

            display_count = min(20, len(normalized_trades))
            display_table(normalized_trades[-display_count:],
                         f"Last {display_count} trades for {args.symbol}")

            # Show additional statistics
            if normalized_trades:
                # Calculate trade statistics
                buy_trades = [t for t in normalized_trades if t['side'] == 'buy']
                sell_trades = [t for t in normalized_trades if t['side'] == 'sell']

                buy_volume = sum(t['amount'] for t in buy_trades)
                sell_volume = sum(t['amount'] for t in sell_trades)

                logger.info(f"\nTrade Statistics:")
                logger.info(f"Buy trades: {len(buy_trades)}, Volume: {buy_volume:.6f}")
                logger.info(f"Sell trades: {len(sell_trades)}, Volume: {sell_volume:.6f}")

                if buy_volume + sell_volume > 0:
                    buy_percentage = (buy_volume / (buy_volume + sell_volume)) * 100
                    logger.info(f"Buy volume percentage: {buy_percentage:.2f}%")

                # Price range
                if normalized_trades:
                    prices = [t['price'] for t in normalized_trades]
                    logger.info(f"Price range: {min(prices):.8f} - {max(prices):.8f}")

        except Exception as e:
            logger.error(f"Error normalizing trade data: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error fetching trade data: {str(e)}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(traceback.format_exc())


def fetch_funding_rate(exchange_service: ExchangeService, args: argparse.Namespace) -> None:
    """
    Fetch and display funding rate for perpetual contracts.

    Args:
        exchange_service: The exchange service to use
        args: Command line arguments with symbol
    """
    try:
        logger.info(f"Fetching funding rate for {args.symbol}...")

        funding_rate = exchange_service.fetch_funding_rate(symbol=args.symbol)

        if not funding_rate:
            logger.warning(f"No funding rate data returned for {args.symbol}")
            return

        try:
            # Normalize funding rate data
            normalized_rate = normalize_funding_rate(
                funding_rate,
                exchange_service.exchange_id,
                args.symbol
            )

            logger.info(f"\nFunding rate for {args.symbol}:")

            # Format the output nicely
            logger.info(f"Rate: {normalized_rate['funding_rate'] * 100:.6f}%")

            if normalized_rate['funding_datetime']:
                logger.info(f"Next funding: {normalized_rate['next_funding_datetime']}")

            # Calculate annualized rate (assuming 3 fundings per day for perpetuals)
            annual_rate = normalized_rate['funding_rate'] * 3 * 365
            logger.info(f"Annualized rate: {annual_rate * 100:.2f}%")

            # Show full details in table format
            headers = ["Field", "Value"]
            rows = [[k, v] for k, v in normalized_rate.items()]
            logger.info("\nFull funding rate details:")
            logger.info(tabulate(rows, headers=headers, tablefmt="psql"))

        except Exception as e:
            logger.error(f"Error processing funding rate data: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())
                logger.debug(f"Raw funding rate data: {funding_rate}")

    except Exception as e:
        logger.error(f"Error fetching funding rate: {e}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Fetch and display cryptocurrency market data")

    parser.add_argument("--exchange", "-e", type=str, default="kraken",
                       help="Exchange ID (default: kraken)")
    parser.add_argument("--symbol", "-s", type=str, default="BTC/USDT",
                       help="Trading pair symbol or comma-separated list (default: BTC/USDT)")
    parser.add_argument("--timeframe", "-t", type=str, default="1h",
                       help="Timeframe for OHLCV data (default: 1h)")
    parser.add_argument("--limit", "-l", type=int, default=100,
                       help="Limit for number of results (default: 100)")
    parser.add_argument("--depth", "-d", type=int, default=10,
                       help="Depth for order book (default: 10)")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days of historical data to fetch (default: 7)")
    parser.add_argument("--sandbox", action="store_true",
                       help="Use exchange sandbox/testnet mode")
    parser.add_argument("--data-type", "-dt", type=str, choices=["ohlcv", "ticker", "orderbook", "trades", "funding"],
                       default="ohlcv", help="Type of data to fetch (default: ohlcv)")

    args = parser.parse_args()

    # exchange_credentials = load_exchange_api_keys()

    try:
        # api_key = None
        # secret = None
        # password = None

        # if exchange_id in exchange_credentials:
        #     creds = exchange_credentials[exchange_id]
        #     api_key = creds.get("api_key")
        #     secret = creds.get("secret")
        #     password = creds.get("password")
        #     logger.info(f"Using authentication for {exchange_id}")
        # else:
        #     logger.info(f"No authentication found for {exchange_id}, using public API only")

        exchange_service = ExchangeService(
            exchange_id=args.exchange,
            # api_key=api_key,
            # secret=secret,
            # password=password,
            # sandbox=args.sandbox,
            timeout=30000,  # 30 seconds
            rate_limit_calls_per_second=2.0,  # Be gentle with the API
            max_retries=3
        )

        logger.info(f"Successfully connected to {args.exchange}")

        if args.data_type == "ohlcv":
            fetch_ohlcv_data(exchange_service, args)
        elif args.data_type == "ticker":
            fetch_ticker_data(exchange_service, args)
        elif args.data_type == "orderbook":
            fetch_order_book_data(exchange_service, args)
        elif args.data_type == "trades":
            fetch_trade_data(exchange_service, args)
        elif args.data_type == "funding":
            fetch_funding_rate(exchange_service, args)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up
        if 'exchange_service' in locals():
            exchange_service.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()