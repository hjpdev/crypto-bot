#!/usr/bin/env python3
"""
Market Analysis Script.

This script fetches market data for selected symbols, runs market analysis,
and displays results in a readable format with visualizations.
"""

import argparse
import sys
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Add the project directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.exchange_service import ExchangeService
from app.services.market_analyzer import MarketAnalyzer
from app.services.market_sentiment import MarketSentiment
from app.services.data_preparation import ohlcv_to_dataframe


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_analysis')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze market conditions for trading.')

    parser.add_argument('--symbols', '-s', nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                        help='Symbols to analyze (default: BTC/USDT ETH/USDT)')
    parser.add_argument('--timeframes', '-t', nargs='+', default=['1h', '4h', '1d'],
                        help='Timeframes to analyze (default: 1h 4h 1d)')
    parser.add_argument('--exchange', '-e', default='binance',
                        help='Exchange to fetch data from (default: binance)')
    parser.add_argument('--lookback', '-l', type=int, default=100,
                        help='Number of candles to analyze (default: 100)')
    parser.add_argument('--output', '-o', default=None,
                        help='Directory to save output files (default: None)')
    parser.add_argument('--correlation', '-c', action='store_true',
                        help='Calculate correlation matrix between symbols')
    parser.add_argument('--regime', '-r', action='store_true',
                        help='Detect market regime for each symbol')
    parser.add_argument('--sentiment', action='store_true',
                        help='Calculate market sentiment')
    parser.add_argument('--levels', action='store_true',
                        help='Identify support and resistance levels')
    parser.add_argument('--volatility', '-v', action='store_true',
                        help='Calculate volatility metrics')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run all analysis types')

    return parser.parse_args()


def fetch_market_data(exchange_service: ExchangeService,
                      symbols: List[str],
                      timeframes: List[str],
                      lookback: int) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for multiple symbols and timeframes.

    Args:
        exchange_service: Exchange service to fetch data from
        symbols: List of symbols to fetch data for
        timeframes: List of timeframes to fetch data for
        lookback: Number of candles to fetch

    Returns:
        Dict mapping symbol to dict of timeframe -> dataframe
    """
    logger.info(f"Fetching market data for {len(symbols)} symbols and {len(timeframes)} timeframes")

    market_data = {}

    for symbol in symbols:
        market_data[symbol] = {}

        for timeframe in timeframes:
            logger.info(f"Fetching {symbol} {timeframe} data...")

            try:
                # Fetch OHLCV data
                ohlcv = exchange_service.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=lookback
                )

                # Convert to DataFrame
                df = ohlcv_to_dataframe(ohlcv)

                # Store in market_data
                market_data[symbol][timeframe] = df

                logger.info(f"  Fetched {len(df)} candles for {symbol} {timeframe}")

            except Exception as e:
                logger.error(f"Error fetching {symbol} {timeframe}: {str(e)}")
                continue

    return market_data


def analyze_markets(market_data: Dict[str, Dict[str, pd.DataFrame]],
                   args: argparse.Namespace) -> Dict[str, Dict]:
    """
    Run market analysis on the fetched data.

    Args:
        market_data: Dict mapping symbol to dict of timeframe -> dataframe
        args: Command-line arguments

    Returns:
        Dict containing analysis results
    """
    logger.info("Analyzing market data...")

    # Initialize analysis services
    market_analyzer = MarketAnalyzer()
    market_sentiment = MarketSentiment()

    # Store results
    results = {}

    # Collect all symbols data for market breadth and correlation analysis
    if args.sentiment or args.correlation or args.all:
        all_symbols_1h_data = {}
        for symbol, timeframes in market_data.items():
            if '1h' in timeframes:
                all_symbols_1h_data[symbol] = timeframes['1h']

    # Calculate market breadth if needed
    breadth_data = None
    if (args.sentiment or args.all) and len(all_symbols_1h_data) > 1:
        logger.info("Calculating market breadth...")
        breadth_data = market_sentiment.get_market_breadth(all_symbols_1h_data)

    # Calculate correlation matrix if requested
    correlation_matrix = None
    if args.correlation or args.all:
        logger.info("Calculating correlation matrix...")
        correlation_matrix = market_analyzer.calculate_correlation_matrix(all_symbols_1h_data)

        # Store in results
        results['correlation_matrix'] = correlation_matrix

    # Analyze each symbol
    for symbol, timeframes in market_data.items():
        logger.info(f"Analyzing {symbol}...")
        results[symbol] = {}

        # Multi-timeframe analysis
        if args.all:
            logger.info(f"  Running full market context analysis...")
            results[symbol]['market_context'] = market_analyzer.get_market_context(symbol, timeframes)

        # Individual analysis types
        else:
            # Analyze each timeframe
            for timeframe, df in timeframes.items():
                results[symbol][timeframe] = {}

                # Detect market regime
                if args.regime or args.all:
                    logger.info(f"  Detecting market regime for {timeframe}...")
                    regime = market_analyzer.detect_market_regime(df, timeframe)
                    results[symbol][timeframe]['regime'] = regime

                # Identify support/resistance levels
                if args.levels or args.all:
                    logger.info(f"  Identifying support/resistance levels for {timeframe}...")
                    levels = market_analyzer.identify_support_resistance(df)
                    results[symbol][timeframe]['support_resistance'] = levels

                # Calculate volatility
                if args.volatility or args.all:
                    logger.info(f"  Calculating volatility for {timeframe}...")
                    volatility = market_analyzer.calculate_volatility(df)
                    results[symbol][timeframe]['volatility'] = volatility

        # Calculate sentiment
        if args.sentiment or args.all:
            logger.info(f"  Calculating market sentiment...")
            # Use 1h timeframe for sentiment by default
            if '1h' in timeframes:
                sentiment = market_sentiment.get_overall_sentiment(
                    symbol,
                    timeframes['1h'],
                    market_breadth_data=breadth_data
                )
                results[symbol]['sentiment'] = sentiment

    return results


def display_results(market_data: Dict[str, Dict[str, pd.DataFrame]],
                   results: Dict[str, Dict],
                   args: argparse.Namespace):
    """
    Display analysis results in a readable format.

    Args:
        market_data: Dict mapping symbol to dict of timeframe -> dataframe
        results: Dict containing analysis results
        args: Command-line arguments
    """
    logger.info("Displaying analysis results...")

    # Display correlation matrix if available
    if 'correlation_matrix' in results:
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['correlation_matrix'], annot=True, cmap='coolwarm', center=0)
        plt.title('Asset Correlation Matrix')
        plt.tight_layout()

        if args.output:
            plt.savefig(os.path.join(args.output, 'correlation_matrix.png'))
        else:
            plt.show()

    # Initialize market analyzer for visualizations
    market_analyzer = MarketAnalyzer()

    # Display results for each symbol
    for symbol in results:
        if symbol == 'correlation_matrix':
            continue

        print(f"\n{'=' * 50}")
        print(f" {symbol} Analysis Summary ")
        print(f"{'=' * 50}")

        # Display sentiment if available
        if 'sentiment' in results[symbol]:
            sentiment = results[symbol]['sentiment']
            print(f"\nSentiment: {sentiment['sentiment_category']} ({sentiment['overall_sentiment_score']:.2f})")
            print(f"Confidence: {sentiment['confidence']:.2f}")

            if 'factor_breakdown' in sentiment:
                print("\nSentiment Factor Breakdown:")
                for factor, score in sentiment['factor_breakdown'].items():
                    print(f"  {factor}: {score:.2f}")

        # Display market context or individual timeframe results
        if 'market_context' in results[symbol]:
            for timeframe, context in results[symbol]['market_context'].items():
                print(f"\n{timeframe} Market Context:")
                print(f"  Regime: {context['regime']}")

                if 'volatility' in context:
                    print(f"  Historical Volatility: {context['volatility'].get('historical_volatility', 0):.2%}")

                if 'support_resistance' in context:
                    support = context['support_resistance'].get('support', [])
                    resistance = context['support_resistance'].get('resistance', [])
                    print(f"  Support Levels: {', '.join(f'{level:.2f}' for level in support[:3])}")
                    print(f"  Resistance Levels: {', '.join(f'{level:.2f}' for level in resistance[:3])}")

                # Create visualization
                if symbol in market_data and timeframe in market_data[symbol]:
                    fig = market_analyzer.visualize_market_analysis(
                        market_data[symbol][timeframe],
                        context,
                        title=f"{symbol} {timeframe} Analysis"
                    )

                    if args.output:
                        plt.savefig(os.path.join(args.output, f"{symbol.replace('/', '_')}_{timeframe}_analysis.png"))
                    else:
                        plt.show()
        else:
            # Display individual analysis results
            for timeframe in results[symbol]:
                if timeframe == 'sentiment':
                    continue

                print(f"\n{timeframe} Analysis:")
                timeframe_results = results[symbol][timeframe]

                if 'regime' in timeframe_results:
                    print(f"  Market Regime: {timeframe_results['regime']}")

                if 'support_resistance' in timeframe_results:
                    support = timeframe_results['support_resistance'].get('support', [])
                    resistance = timeframe_results['support_resistance'].get('resistance', [])
                    print(f"  Support Levels: {', '.join(f'{level:.2f}' for level in support[:3])}")
                    print(f"  Resistance Levels: {', '.join(f'{level:.2f}' for level in resistance[:3])}")

                if 'volatility' in timeframe_results:
                    vol = timeframe_results['volatility']
                    print(f"  Historical Volatility: {vol.get('historical_volatility', 0):.2%}")
                    print(f"  ATR Volatility: {vol.get('atr_volatility', 0):.2%}")


def main():
    """Run the market analysis script."""
    # Parse command-line arguments
    args = parse_arguments()

    # If no specific analysis is selected, default to running all
    if not any([args.regime, args.sentiment, args.levels, args.volatility, args.correlation, args.all]):
        args.all = True

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Initialize services
    exchange_service = ExchangeService(args.exchange)

    try:
        # Fetch market data
        market_data = fetch_market_data(
            exchange_service=exchange_service,
            symbols=args.symbols,
            timeframes=args.timeframes,
            lookback=args.lookback
        )

        # Run analysis
        results = analyze_markets(market_data, args)

        # Display results
        display_results(market_data, results, args)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

    logger.info("Market analysis completed successfully")


if __name__ == "__main__":
    main()