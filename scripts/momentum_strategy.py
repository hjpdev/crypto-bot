#!/usr/bin/env python3
"""
Momentum Strategy Test Script.

This script demonstrates the functionality of the MomentumStrategy by:
1. Initializing the strategy with sample configuration
2. Fetching historical market data for selected symbols
3. Generating trading signals
4. Visualizing the signals on price charts
5. Calculating performance metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.strategies.momentum_strategy import MomentumStrategy
from app.services.indicator_service import IndicatorService
from app.models.signals import SignalType

# Set up styling for plots
sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


def fetch_sample_data(symbol, timeframe="1h", days=30):
    """
    Fetch or generate sample market data.

    In a real scenario, this would connect to an exchange API or database.
    For this test script, we generate synthetic data.

    Args:
        symbol: The trading symbol
        timeframe: The timeframe for the data
        days: Number of days of historical data to generate

    Returns:
        A dictionary with market data
    """
    print(f"Generating sample data for {symbol} on {timeframe} timeframe...")

    periods = days * 24  # For hourly data
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='h')

    # Base price varies by symbol
    if symbol == "BTC/USDT":
        base_price = 50000
        volatility = 1000
    elif symbol == "ETH/USDT":
        base_price = 3000
        volatility = 100
    else:
        base_price = 100
        volatility = 10

    # Generate price data with some randomness and trends
    np.random.seed(42 + hash(symbol) % 100)  # Different seed for each symbol

    # Randomly select one of the trend patterns
    trend_pattern = np.random.choice([0, 1, 2])

    # Apply the selected trend pattern
    if trend_pattern == 0:
        # Uptrend
        trend = np.linspace(0, 1, periods)
    elif trend_pattern == 1:
        # Downtrend
        trend = np.linspace(1, 0, periods)
    else:
        # Cyclical
        trend = np.sin(np.linspace(0, 4*np.pi, periods))

    # Add random walk component
    random_walk = np.random.randn(periods).cumsum() * (volatility / 10)

    # Combine for final price series
    close = base_price + (trend * volatility) + random_walk

    # Generate OHLCV data
    high = close * (1 + np.random.rand(periods) * 0.02)
    low = close * (1 - np.random.rand(periods) * 0.02)
    open_price = low + (high - low) * np.random.rand(periods)
    volume = (base_price * 100 + np.random.randn(periods).cumsum() * 1000).clip(min=base_price*50)

    # Create specific patterns near the end for signal generation
    # Last 5 days: create a trend reversal
    pattern_len = 5 * 24

    # Save the last value before the pattern to ensure a smooth transition
    last_value_before_pattern = close[-pattern_len]

    if np.random.rand() > 0.5:
        # Bearish to bullish reversal with randomness
        # Create a base curve that's sigmoidal rather than linear
        x = np.linspace(-3, 3, pattern_len)
        sigmoid = 1 / (1 + np.exp(-x))  # Sigmoid function for a more natural curve

        # Scale the sigmoid to the desired price range
        start_pct = 0.95  # Starting at 5% below base
        end_pct = 1.10    # Ending at 10% above base

        # Scale to the price movement range
        scaled_curve = start_pct + sigmoid * (end_pct - start_pct)

        # Add some noise to make it look more natural
        noise = np.random.randn(pattern_len) * 0.003  # 0.3% noise
        pattern = (scaled_curve + noise) * base_price

        # Ensure the pattern starts at the last value for continuity
        pattern = pattern - pattern[0] + last_value_before_pattern

        close[-pattern_len:] = pattern

        # Increase volume near reversal with some randomness
        volume_multiplier = 1.5 + np.random.rand(pattern_len//2) * 1.0  # 1.5x to 2.5x increase
        volume[-pattern_len//2:] = volume[-pattern_len//2:] * volume_multiplier
    else:
        # Bullish to bearish reversal with randomness
        # Create a base curve that's sigmoidal rather than linear
        x = np.linspace(3, -3, pattern_len)  # Reversed for bearish
        sigmoid = 1 / (1 + np.exp(-x))  # Sigmoid function for a more natural curve

        # Scale the sigmoid to the desired price range
        start_pct = 1.05  # Starting at 5% above base
        end_pct = 0.95    # Ending at 5% below base

        # Scale to the price movement range
        scaled_curve = start_pct - sigmoid * (start_pct - end_pct)

        # Add some noise to make it look more natural
        noise = np.random.randn(pattern_len) * 0.003  # 0.3% noise
        pattern = (scaled_curve + noise) * base_price

        # Ensure the pattern starts at the last value for continuity
        pattern = pattern - pattern[0] + last_value_before_pattern

        close[-pattern_len:] = pattern

        # Increase volume near reversal with some randomness
        volume_multiplier = 1.5 + np.random.rand(pattern_len//2) * 1.0  # 1.5x to 2.5x increase
        volume[-pattern_len//2:] = volume[-pattern_len//2:] * volume_multiplier

    # Create DataFrame with the data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ohlcv_data": df
    }


def visualize_signals(market_data, signals, output_dir=None):
    """
    Visualize the market data with trading signals.

    Args:
        market_data: Dictionary with market data
        signals: List of signal dictionaries
        output_dir: Directory to save the plots to (optional)
    """
    symbol = market_data["symbol"]
    timeframe = market_data["timeframe"]
    df = market_data["ohlcv_data"].copy()

    # Ensure dataframe has proper index
    if df.index.name != 'timestamp' and 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Create a figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(14, 10))
    fig.suptitle(f'Momentum Strategy Analysis - {symbol} ({timeframe})', fontsize=16)

    # Plot price chart on main axis
    ax1.plot(df.index, df['close'], label='Close Price', color='black', alpha=0.7)

    # Add EMA line to price chart
    if 'ema' in df.columns:
        ax1.plot(df.index, df['ema'], label=f'EMA', color='blue', alpha=0.7)

    # Plot the buy signals
    buy_signals = [s for s in signals if s["type"] == SignalType.BUY]
    if buy_signals:
        # Convert decimal prices to float for plotting
        buy_prices = [float(s["price"]) for s in buy_signals]
        buy_confidences = [s["confidence"] for s in buy_signals]

        # Use the last few indices for signal placement (simplified approach)
        # In a real scenario, we would map timestamps more precisely
        latest_indices = df.index[-len(buy_signals):] if len(buy_signals) <= len(df) else df.index[-1:]

        # Scale marker size by confidence
        sizes = [max(100 * conf, 50) for conf in buy_confidences]

        # Plot buy signals
        ax1.scatter(latest_indices, buy_prices, color='green', s=sizes,
                   marker='^', label='Buy Signal', alpha=0.8)

    # Plot the sell signals
    sell_signals = [s for s in signals if s["type"] == SignalType.SELL]
    if sell_signals:
        # Convert decimal prices to float for plotting
        sell_prices = [float(s["price"]) for s in sell_signals]
        sell_confidences = [s["confidence"] for s in sell_signals]

        # Use the last few indices for signal placement (simplified approach)
        latest_indices = df.index[-len(sell_signals):] if len(sell_signals) <= len(df) else df.index[-1:]

        # Scale marker size by confidence
        sizes = [max(100 * conf, 50) for conf in sell_confidences]

        # Plot sell signals
        ax1.scatter(latest_indices, sell_prices, color='red', s=sizes,
                   marker='v', label='Sell Signal', alpha=0.8)

    # Plot RSI on second axis if available
    if 'rsi' in df.columns:
        ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI')
        ax2.legend(loc='upper left')

    # Plot MACD on third axis if available
    if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
        ax3.plot(df.index, df['MACD_12_26_9'], label='MACD', color='blue')
        ax3.plot(df.index, df['MACDs_12_26_9'], label='Signal', color='red')

        # Plot histogram
        if 'MACDh_12_26_9' in df.columns:
            histogram = df['MACDh_12_26_9']
            positive = histogram > 0
            negative = histogram < 0

            ax3.bar(df.index[positive], histogram[positive], color='green', alpha=0.5, width=0.5)
            ax3.bar(df.index[negative], histogram[negative], color='red', alpha=0.5, width=0.5)

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax3.set_ylabel('MACD')
        ax3.legend(loc='upper left')

    # Format the charts
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Focus the x-axis on the last 30% of the data where signals are
    start_idx = int(len(df) * 0.7)
    xmin, xmax = df.index[start_idx], df.index[-1]
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    ax3.set_xlim(xmin, xmax)

    # Add a timestamp to the plot
    plt.figtext(0.01, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='left', fontsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the plot if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol.replace('/', '_')}_{timeframe}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_dir / filename}")

    # Show the plot
    plt.show()


def print_signal_details(signals):
    """Print detailed information about the generated signals."""
    if not signals:
        print("No signals were generated.")
        return

    print(f"\n{'=' * 50}")
    print(f"SIGNAL DETAILS ({len(signals)} signals)")
    print(f"{'=' * 50}")

    for i, signal in enumerate(signals, 1):
        print(f"\nSIGNAL {i}:")
        print(f"  Symbol:     {signal['symbol']}")
        print(f"  Type:       {signal['type']}")
        print(f"  Price:      {signal['price']}")
        print(f"  Confidence: {signal['confidence']:.2f}")
        print(f"  Timeframe:  {signal['timeframe']}")

        print("  Indicators:")
        for name, value in signal['indicators'].items():
            print(f"    {name}: {value:.2f}")

        print("  Conditions Met:")
        for name, met in signal['metadata']['conditions'].items():
            print(f"    {name}: {'✓' if met else '✗'}")


def calculate_performance_metrics(market_data, signals, days_forward=5):
    """
    Calculate performance metrics for the signals.

    Args:
        market_data: Dictionary with market data
        signals: List of signal dictionaries
        days_forward: Number of days forward to analyze performance

    Returns:
        Dictionary with performance metrics
    """
    df = market_data["ohlcv_data"].copy()

    # Ensure dataframe has proper index
    if df.index.name != 'timestamp' and 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    if not signals:
        return {"error": "No signals to evaluate"}

    print(f"\n{'=' * 50}")
    print(f"PERFORMANCE ANALYSIS (Looking {days_forward} days forward)")
    print(f"{'=' * 50}")

    # Get the buy and sell signals
    buy_signals = [s for s in signals if s["type"] == SignalType.BUY]
    sell_signals = [s for s in signals if s["type"] == SignalType.SELL]

    results = {
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "buy_success_rate": 0,
        "sell_success_rate": 0,
        "avg_profit_percentage": 0,
        "max_profit_percentage": 0,
        "max_loss_percentage": 0
    }

    # Analyze buy signals (success = price goes up)
    if buy_signals:
        successful_buys = 0
        profits = []

        for signal in buy_signals:
            # Use last rows for demonstration since signals would be recent
            signal_price = float(signal["price"])

            # Find closest price in the dataframe
            closest_idx = df['close'].sub(signal_price).abs().idxmin()
            idx_pos = df.index.get_loc(closest_idx)

            # Get future prices (limited by available data)
            future_periods = min(days_forward * 24, len(df) - idx_pos - 1)
            if future_periods <= 0:
                continue

            future_idx = df.index[idx_pos + future_periods]
            future_price = df.loc[future_idx, 'close']

            # Calculate profit/loss percentage
            pnl_pct = (future_price - signal_price) / signal_price * 100
            profits.append(pnl_pct)

            if pnl_pct > 0:
                successful_buys += 1

            print(f"Buy  signal at {signal_price:.2f}: Future price after {future_periods//24} days: {future_price:.2f} | P/L: {pnl_pct:+.2f}%")

        if profits:
            results["buy_success_rate"] = successful_buys / len(buy_signals) * 100
            results["avg_buy_profit"] = sum(profits) / len(profits)
            results["max_buy_profit"] = max(profits) if profits else 0
            results["max_buy_loss"] = min(profits) if profits else 0

    # Analyze sell signals (success = price goes down)
    if sell_signals:
        successful_sells = 0
        profits = []

        for signal in sell_signals:
            # Use last rows for demonstration since signals would be recent
            signal_price = float(signal["price"])

            # Find closest price in the dataframe
            closest_idx = df['close'].sub(signal_price).abs().idxmin()
            idx_pos = df.index.get_loc(closest_idx)

            # Get future prices (limited by available data)
            future_periods = min(days_forward * 24, len(df) - idx_pos - 1)
            if future_periods <= 0:
                continue

            future_idx = df.index[idx_pos + future_periods]
            future_price = df.loc[future_idx, 'close']

            # For sell signals, profit if price goes down
            pnl_pct = (signal_price - future_price) / signal_price * 100
            profits.append(pnl_pct)

            if pnl_pct > 0:
                successful_sells += 1

            print(f"Sell signal at {signal_price:.2f}: Future price after {future_periods//24} days: {future_price:.2f} | P/L: {pnl_pct:+.2f}%")

        if profits:
            results["sell_success_rate"] = successful_sells / len(sell_signals) * 100
            results["avg_sell_profit"] = sum(profits) / len(profits)
            results["max_sell_profit"] = max(profits) if profits else 0
            results["max_sell_loss"] = min(profits) if profits else 0

    # Print summary
    print("\nPerformance Summary:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    return results


def main():
    """Run the momentum strategy test."""
    print("Momentum Strategy Test")
    print("=====================\n")

    # Create configuration for the strategy
    config = {
        "risk_per_trade": 2.0,
        "max_open_positions": 5,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "volume_change_threshold": 1.5,
        "trend_ema_period": 50,
        "min_confidence_threshold": 0.6,
        "atr_multiplier": 2.0,
        "risk_reward_targets": [1.5, 2.5, 3.5],
        "timeframes": ["1h", "4h", "1d"]
    }

    # Initialize the strategy
    print("Initializing Momentum Strategy with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("")

    strategy = MomentumStrategy(config)

    # Define the symbols to test with
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    # Create output directory for charts
    output_dir = Path("./output/momentum_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each symbol
    all_signals = []

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")

        # Fetch sample data
        market_data = fetch_sample_data(symbol, "1h", days=30)

        # Calculate indicators (these would normally come from the indicator service)
        df = market_data["ohlcv_data"]
        # Set timestamp as index if it's not already
        if 'timestamp' in df.columns and df.index.name != 'timestamp':
            df = df.set_index('timestamp')

        df = IndicatorService.calculate_rsi(df, period=config["rsi_period"])
        df = IndicatorService.calculate_macd(
            df,
            fast=config["macd_fast"],
            slow=config["macd_slow"],
            signal=config["macd_signal"]
        )
        df = IndicatorService.calculate_ema(df, period=config["trend_ema_period"])
        df = IndicatorService.calculate_atr(df)
        market_data["ohlcv_data"] = df

        # Generate signals
        print(f"Generating signals for {symbol}...")
        signals = strategy.generate_signals(market_data)
        all_signals.extend(signals)

        # Print signal details
        print_signal_details(signals)

        # Calculate performance metrics
        calculate_performance_metrics(market_data, signals)

        # Visualize the signals
        visualize_signals(market_data, signals, output_dir)

    print(f"\nTotal signals generated across all symbols: {len(all_signals)}")
    print(f"Buy signals: {len([s for s in all_signals if s['type'] == SignalType.BUY])}")
    print(f"Sell signals: {len([s for s in all_signals if s['type'] == SignalType.SELL])}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()