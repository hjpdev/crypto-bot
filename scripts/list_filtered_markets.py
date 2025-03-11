#!/usr/bin/env python
"""
Script to list filtered cryptocurrency markets.

This script loads exchange configuration, connects to the exchange,
and runs filtering on the available markets based on the filtering criteria.
It outputs filtered symbols with their properties and shows why specific symbols were filtered out.
"""

import argparse
import logging
import os
import sys
import yaml
from tabulate import tabulate
from typing import Dict, List, Any, Tuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.exchange_service import ExchangeService
from app.services.market_filter import MarketFilter

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def setup_exchange(exchange_id: str, config: Dict[str, Any]) -> ExchangeService:
    return ExchangeService(
        exchange_id=exchange_id,
        # @TODO: Add API key and secret
        # api_key=exchange_config.get("api_key"),
        # secret=exchange_config.get("api_secret"),
        # password=exchange_config.get("password"),
        # sandbox=exchange_config.get("test_mode", True),
        timeout=30000,
        enableRateLimit=True,
    )


def filter_markets(exchange_service: ExchangeService, filter_config: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    market_filter = MarketFilter(exchange_service)
    rejection_reasons = {}

    # Get all available symbols
    logger.info("Fetching available markets...")
    try:
        symbols = exchange_service.exchange.symbols
        logger.info(f"Found {len(symbols)} markets on {exchange_service.exchange_id}")
    except Exception as e:
        logger.error(f"Error fetching markets: {str(e)}")
        return [], {}

    # Keep track of all symbols that get filtered out and why
    initial_symbols = symbols.copy()

    # Apply all filters at once using the MarketFilter service
    filtered_symbols = market_filter.apply_all_filters(symbols, filter_config)

    for filter_type in ['quote', 'market_cap', 'volume', 'spread', 'volatility']:
        # Skip filters not specified in the config
        if filter_type == 'quote' and 'allowed_quotes' not in filter_config:
            continue
        if filter_type == 'market_cap' and ('min_market_cap' not in filter_config or filter_config['min_market_cap'] is None):
            continue
        if filter_type == 'volume' and ('min_volume' not in filter_config or filter_config['min_volume'] is None):
            continue
        if filter_type == 'spread' and ('max_spread' not in filter_config or filter_config['max_spread'] is None):
            continue
        if filter_type == 'volatility' and (
            'min_volatility' not in filter_config or filter_config['min_volatility'] is None or
            'max_volatility' not in filter_config or filter_config['max_volatility'] is None):
            continue

        # Use the MarketFilter's individual filter functions to determine rejected symbols
        if filter_type == 'quote':
            remaining = market_filter.filter_by_allowed_quote(initial_symbols, filter_config['allowed_quotes'])
            rejected = set(initial_symbols) - set(remaining)
            if rejected:
                rejection_reasons['quote'] = list(rejected)
                initial_symbols = remaining
        elif filter_type == 'market_cap':
            remaining = market_filter.filter_by_market_cap(initial_symbols, filter_config['min_market_cap'])
            rejected = set(initial_symbols) - set(remaining)
            if rejected:
                rejection_reasons['market_cap'] = list(rejected)
                initial_symbols = remaining
        elif filter_type == 'volume':
            remaining = market_filter.filter_by_volume(initial_symbols, filter_config['min_volume'])
            rejected = set(initial_symbols) - set(remaining)
            if rejected:
                rejection_reasons['volume'] = list(rejected)
                initial_symbols = remaining
        elif filter_type == 'spread':
            remaining = market_filter.filter_by_spread(initial_symbols, filter_config['max_spread'])
            rejected = set(initial_symbols) - set(remaining)
            if rejected:
                rejection_reasons['spread'] = list(rejected)
                initial_symbols = remaining
        elif filter_type == 'volatility':
            remaining = market_filter.filter_by_volatility(
                initial_symbols,
                filter_config['min_volatility'],
                filter_config['max_volatility'],
                filter_config.get('volatility_timeframe', '1d'),
                filter_config.get('volatility_periods', 14)
            )
            rejected = set(initial_symbols) - set(remaining)
            if rejected:
                rejection_reasons['volatility'] = list(rejected)
                initial_symbols = remaining

    return filtered_symbols, rejection_reasons


def get_market_properties(exchange: ExchangeService, symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Get properties for the provided symbols.

    Args:
        exchange: ExchangeService to use
        symbols: List of symbols to get properties for

    Returns:
        List of dictionaries with market properties
    """
    market_filter = MarketFilter(exchange)
    market_data = market_filter._get_market_data(symbols)

    # Get volatility data for the symbols
    volatility_data = market_filter._get_volatility_data(symbols)

    # Compile properties
    properties = []
    for symbol in symbols:
        if symbol in market_data:
            ticker = market_data[symbol]

            # Get values or defaults
            last_price = ticker.get('last', 'N/A')
            volume = None
            for field in ['quoteVolume', 'quote_volume', 'volume']:
                if field in ticker and ticker[field] is not None:
                    volume = ticker[field]
                    break

            market_cap = None
            for field in ['marketCap', 'market_cap', 'cap']:
                if field in ticker:
                    market_cap = ticker[field]
                    break

            # Get spread
            spread = ticker.get('spread', 'N/A')

            # Get volatility
            volatility = volatility_data.get(symbol, 'N/A')

            # Add to properties list
            properties.append({
                'symbol': symbol,
                'price': last_price,
                'volume_24h': volume,
                'market_cap': market_cap,
                'spread': spread,
                'volatility': volatility
            })

    return properties


def format_number(value: Any) -> str:
    """Format number with appropriate units."""
    if value is None or value == 'N/A':
        return 'N/A'

    if isinstance(value, (int, float)):
        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value / 1_000:.2f}K"
        else:
            return f"${value:.2f}"

    return str(value)


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(description='List filtered cryptocurrency markets.')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--exchange', default='binance', help='Exchange ID to use')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--min-market-cap', type=float, help='Minimum market cap (USD)')
    parser.add_argument('--min-volume', type=float, help='Minimum 24h volume (USD)')
    parser.add_argument('--max-spread', type=float, help='Maximum bid-ask spread (%%)')
    parser.add_argument('--min-vol', type=float, help='Minimum volatility (%%)')
    parser.add_argument('--max-vol', type=float, help='Maximum volatility (%%)')
    parser.add_argument('--quotes', nargs='+', help='Allowed quote currencies (e.g. USD USDT)')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return 1

    # Set up exchange
    try:
        exchange = setup_exchange(args.exchange, config)
    except Exception as e:
        logging.error(f"Error setting up exchange: {str(e)}")
        return 1

    # Create filter config - combine CLI args with config file
    # CLI args take precedence if provided
    filter_config = {}

    # Get settings from market_filter section of config if it exists
    if 'market_filter' in config:
        filter_config.update(config['market_filter'])

    # Override with CLI arguments if provided
    if args.min_market_cap is not None:
        filter_config['min_market_cap'] = args.min_market_cap

    if args.min_volume is not None:
        filter_config['min_volume'] = args.min_volume

    if args.max_spread is not None:
        filter_config['max_spread'] = args.max_spread

    if args.min_vol is not None:
        filter_config['min_volatility'] = args.min_vol

    if args.max_vol is not None:
        filter_config['max_volatility'] = args.max_vol

    if args.quotes:
        filter_config['allowed_quotes'] = args.quotes

    # Filter markets
    try:
        filtered_symbols, rejection_reasons = filter_markets(exchange, filter_config)
    except Exception as e:
        logging.error(f"Error filtering markets: {str(e)}")
        return 1

    # Get properties for filtered markets
    try:
        market_properties = get_market_properties(exchange, filtered_symbols)
    except Exception as e:
        logging.error(f"Error getting market properties: {str(e)}")
        return 1

    # Display results
    print(f"\nFiltered Markets on {args.exchange.upper()}")
    print(f"-------------------------------")
    print(f"Total filtered symbols: {len(filtered_symbols)}")
    print("\nFilter criteria:")
    for key, value in filter_config.items():
        print(f"  {key}: {value}")

    if market_properties:
        print("\nFiltered Markets:")

        # Format table data
        table_data = []
        for market in market_properties:
            table_data.append([
                market['symbol'],
                format_number(market['price']),
                format_number(market['volume_24h']),
                format_number(market['market_cap']),
                f"{market['spread']}%" if market['spread'] != 'N/A' else 'N/A',
                f"{market['volatility']}%" if market['volatility'] != 'N/A' else 'N/A'
            ])

        headers = ["Symbol", "Price", "Volume (24h)", "Market Cap", "Spread", "Volatility"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Show rejection reasons
    if args.verbose and rejection_reasons:
        print("\nRejection Reasons:")
        for reason, symbols in rejection_reasons.items():
            print(f"\n{reason.replace('_', ' ').title()}:")
            for i, symbol in enumerate(sorted(symbols)):
                print(f"  {symbol}", end=", " if (i + 1) % 5 != 0 else "\n")
            if len(symbols) % 5 != 0:
                print()  # Ensure newline at end

    return 0


if __name__ == "__main__":
    sys.exit(main())