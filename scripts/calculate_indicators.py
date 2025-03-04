#!/usr/bin/env python3
"""
Script to demonstrate the usage of the indicator service.

This script fetches market data for a specified symbol, calculates
a comprehensive set of technical indicators, and displays or plots the results.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to sys.path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.services.indicator_service import IndicatorService
from app.services.indicator_utils import normalize_indicator


def fetch_sample_data(days: int = 100) -> pd.DataFrame:
    """
    Create sample market data for demonstration purposes.
    In a real implementation, this would fetch data from an exchange.

    Args:
        days: Number of days of data to generate

    Returns:
        DataFrame with OHLCV data
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate sample price data with a trend and some noise
    np.random.seed(42)  # For reproducibility
    base_price = 10000  # Starting price
    trend = np.linspace(0, 5, len(dates))  # Upward trend
    noise = np.random.normal(0, 1, len(dates))  # Random noise
    cycle = 5 * np.sin(np.linspace(0, 15, len(dates)))  # Cyclic component

    # Combine components to create the price movement
    close_prices = base_price + (100 * trend) + (50 * noise) + (100 * cycle)

    # Generate other OHLCV data based on close prices
    data = {
        'open': [close_prices[i-1] if i > 0 else close_prices[0] for i in range(len(dates))],
        'high': [p + abs(20 * noise[i]) for i, p in enumerate(close_prices)],
        'low': [p - abs(20 * noise[i]) for i, p in enumerate(close_prices)],
        'close': close_prices,
        'volume': [1000000 + 500000 * abs(noise[i]) + 100000 * trend[i] for i in range(len(dates))]
    }

    df = pd.DataFrame(data, index=dates)
    return df


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a comprehensive set of indicators on the provided data.

    Args:
        data: OHLCV DataFrame

    Returns:
        DataFrame with calculated indicators
    """
    # Define indicators configuration for batch calculation
    indicators_config = {
        'rsi': {'period': 14},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'ema': [{'period': 8}, {'period': 21}, {'period': 55}],
        'sma': [{'period': 20}, {'period': 50}, {'period': 200}],
        'bollinger_bands': {'period': 20, 'std_dev': 2},
        'atr': {'period': 14},
        'adx': {'period': 14},
        'obv': {},
        'vwap': {},
        'support_resistance': {'lookback': 14}
    }

    # Calculate all indicators using batch_calculate
    result = IndicatorService.batch_calculate(data, indicators_config)

    # Calculated one by one for demonstration
    print("Calculating indicators individually...")

    # RSI
    print("- Calculating RSI...")
    rsi_df = IndicatorService.calculate_rsi(data, period=14)

    # Bollinger Bands
    print("- Calculating Bollinger Bands...")
    bb_df = IndicatorService.calculate_bollinger_bands(data, period=20, std_dev=2)

    # ATR
    print("- Calculating ATR...")
    atr_df = IndicatorService.calculate_atr(data, period=14)

    # ADX
    print("- Calculating ADX...")
    adx_df = IndicatorService.calculate_adx(data, period=14)

    # OBV
    print("- Calculating OBV...")
    obv_df = IndicatorService.calculate_obv(data)

    # VWAP
    print("- Calculating VWAP...")
    vwap_df = IndicatorService.calculate_vwap(data)

    # Support/Resistance
    print("- Calculating Support/Resistance levels...")
    sr_df = IndicatorService.calculate_support_resistance(data, lookback=14)

    # Detect divergence between price and RSI
    print("- Detecting divergence between price and RSI...")
    divergence_df = IndicatorService.detect_divergence(
        data['close'],
        rsi_df['rsi'].fillna(50),  # Fill NaN values for demonstration
        window=10
    )

    # Add divergence information to the result dataframe
    if 'bullish_divergence' in divergence_df.columns:
        result['bullish_divergence'] = divergence_df['bullish_divergence']
    if 'bearish_divergence' in divergence_df.columns:
        result['bearish_divergence'] = divergence_df['bearish_divergence']

    # Calculate multi-timeframe analysis (for demonstration we'll just use the same data)
    print("- Calculating Multi-timeframe RSI...")
    # In a real implementation, we would have data for different timeframes
    # For demonstration, we'll use resampled versions of the same data
    dataframes_dict = {
        '1d': data,
        '3d': data.iloc[::3].copy(),  # Every 3rd row
        '7d': data.iloc[::7].copy()   # Every 7th row
    }

    multi_tf_rsi = IndicatorService.calculate_multi_timeframe(
        dataframes_dict,
        IndicatorService.calculate_rsi,
        period=14
    )

    # Add the multi-timeframe RSI to the result dataframe
    result['rsi_3d'] = pd.Series(index=result.index)
    result['rsi_7d'] = pd.Series(index=result.index)

    # Map the 3d and 7d RSI values to the 1d dataframe
    for date in multi_tf_rsi['3d'].index:
        if date in result.index:
            result.loc[date, 'rsi_3d'] = multi_tf_rsi['3d'].loc[date, 'rsi']

    for date in multi_tf_rsi['7d'].index:
        if date in result.index:
            result.loc[date, 'rsi_7d'] = multi_tf_rsi['7d'].loc[date, 'rsi']

    # Forward fill the multi-timeframe RSI values using ffill() instead of fillna(method='ffill')
    result['rsi_3d'] = result['rsi_3d'].ffill()
    result['rsi_7d'] = result['rsi_7d'].ffill()

    return result


def plot_indicators(data: pd.DataFrame) -> None:
    """
    Plot the calculated indicators for visualization.

    Args:
        data: DataFrame with OHLCV data and calculated indicators
    """
    # Use a style that works with current matplotlib version
    try:
        # Try the new seaborn style naming convention
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            # Fallback to the older style if available
            plt.style.use('seaborn-darkgrid')
        except:
            # If all fails, use the default style
            plt.style.use('default')

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(14, 18))

    # Main price chart with EMAs and Bollinger Bands
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    ax1.set_title('Price with EMAs and Bollinger Bands')
    ax1.plot(data.index, data['close'], label='Close Price')

    if 'ema_8' in data.columns:
        ax1.plot(data.index, data['ema_8'], label='EMA(8)', alpha=0.7)
    if 'ema_21' in data.columns:
        ax1.plot(data.index, data['ema_21'], label='EMA(21)', alpha=0.7)

    if 'BBU_20_2.0' in data.columns and 'BBL_20_2.0' in data.columns:
        ax1.plot(data.index, data['BBU_20_2.0'], 'g--', label='Upper BB', alpha=0.5)
        ax1.plot(data.index, data['BBL_20_2.0'], 'r--', label='Lower BB', alpha=0.5)
        ax1.fill_between(
            data.index,
            data['BBU_20_2.0'],
            data['BBL_20_2.0'],
            alpha=0.1,
            color='gray'
        )

    # Plot support and resistance levels
    if 'support' in data.columns and 'resistance' in data.columns:
        # Only plot non-zero values
        support_points = data[data['support'] > 0]
        resistance_points = data[data['resistance'] > 0]

        if not support_points.empty:
            ax1.scatter(
                support_points.index,
                support_points['support'],
                marker='^',
                color='green',
                s=100,
                label='Support'
            )

        if not resistance_points.empty:
            ax1.scatter(
                resistance_points.index,
                resistance_points['resistance'],
                marker='v',
                color='red',
                s=100,
                label='Resistance'
            )

    # Plot divergence points if present
    if 'bullish_divergence' in data.columns:
        bullish_div_points = data[data['bullish_divergence']].index
        for point in bullish_div_points:
            ax1.axvline(x=point, color='green', linestyle='--', alpha=0.5)

    if 'bearish_divergence' in data.columns:
        bearish_div_points = data[data['bearish_divergence']].index
        for point in bearish_div_points:
            ax1.axvline(x=point, color='red', linestyle='--', alpha=0.5)

    ax1.legend(loc='upper left')
    ax1.grid(True)

    # RSI subplot
    ax2 = plt.subplot2grid((5, 1), (2, 0))
    ax2.set_title('RSI with Multi-timeframe (1d, 3d, 7d)')

    if 'rsi' in data.columns:
        ax2.plot(data.index, data['rsi'], label='RSI (1d)')

        # Add multi-timeframe RSI if available
        if 'rsi_3d' in data.columns:
            ax2.plot(data.index, data['rsi_3d'], label='RSI (3d)', alpha=0.7)
        if 'rsi_7d' in data.columns:
            ax2.plot(data.index, data['rsi_7d'], label='RSI (7d)', alpha=0.7)

        # Add RSI reference lines
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.axhline(y=50, color='k', linestyle='--', alpha=0.3)

        # Set y-limits for RSI
        ax2.set_ylim(0, 100)

    ax2.legend(loc='upper left')
    ax2.grid(True)

    # MACD subplot
    ax3 = plt.subplot2grid((5, 1), (3, 0))
    ax3.set_title('MACD')

    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
        ax3.plot(data.index, data['MACD_12_26_9'], label='MACD')
        ax3.plot(data.index, data['MACDs_12_26_9'], label='Signal')

        # Plot MACD histogram
        if 'MACDh_12_26_9' in data.columns:
            histogram = data['MACDh_12_26_9']
            positive = histogram > 0
            negative = histogram < 0

            ax3.bar(
                data.index[positive],
                histogram[positive],
                color='green',
                alpha=0.5,
                label='Positive'
            )
            ax3.bar(
                data.index[negative],
                histogram[negative],
                color='red',
                alpha=0.5,
                label='Negative'
            )

        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax3.legend(loc='upper left')
    ax3.grid(True)

    # ATR, ADX and Volume subplot
    ax4 = plt.subplot2grid((5, 1), (4, 0))
    ax4.set_title('ATR, ADX, and Volume')

    # Create twin axis for volume
    ax4_volume = ax4.twinx()

    # Plot ATR if available
    if 'atr' in data.columns:
        normalized_atr = normalize_indicator(data['atr'].dropna(), method='minmax') * 100
        ax4.plot(data.index[-len(normalized_atr):], normalized_atr, 'b-', label='ATR (norm)')

    # Plot ADX if available
    if 'ADX_14' in data.columns:
        ax4.plot(data.index, data['ADX_14'], 'r-', label='ADX')

        # Add ADX reference line
        ax4.axhline(y=25, color='r', linestyle='--', alpha=0.5)

    # Plot Volume if available
    if 'volume' in data.columns:
        ax4_volume.bar(
            data.index,
            data['volume'],
            alpha=0.3,
            color='gray',
            label='Volume'
        )

        # Add OBV if available
        if 'obv' in data.columns:
            normalized_obv = normalize_indicator(data['obv'], method='minmax')
            normalized_obv = normalized_obv * (data['volume'].max() / 2)
            ax4_volume.plot(
                data.index,
                normalized_obv,
                'g-',
                label='OBV (scaled)',
                alpha=0.7
            )

    # Set labels and legends
    ax4.set_ylabel('ATR / ADX')
    ax4_volume.set_ylabel('Volume')

    # Create combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_volume.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax4.grid(True)
    ax4.set_xlabel('Date')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def display_results(data: pd.DataFrame) -> None:
    """
    Display a summary of the calculated indicators.

    Args:
        data: DataFrame with OHLCV data and calculated indicators
    """
    # Display summary statistics
    print("\n=== SUMMARY STATISTICS ===")

    # Last row (most recent data)
    last_row = data.iloc[-1]

    # Current price and trend
    print(f"Current Price: ${last_row['close']:.2f}")

    # RSI Analysis
    if 'rsi' in data.columns:
        rsi_value = last_row['rsi']
        print(f"RSI: {rsi_value:.2f}", end=" ")

        if rsi_value > 70:
            print("(Overbought)")
        elif rsi_value < 30:
            print("(Oversold)")
        else:
            print("(Neutral)")

    # MACD Analysis
    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
        macd = last_row['MACD_12_26_9']
        signal = last_row['MACDs_12_26_9']
        hist = last_row['MACDh_12_26_9'] if 'MACDh_12_26_9' in data.columns else macd - signal

        print(f"MACD: {macd:.4f}, Signal: {signal:.4f}, Histogram: {hist:.4f}", end=" ")

        if hist > 0 and hist > data['MACDh_12_26_9'].iloc[-2]:
            print("(Bullish momentum increasing)")
        elif hist > 0 and hist < data['MACDh_12_26_9'].iloc[-2]:
            print("(Bullish momentum weakening)")
        elif hist < 0 and hist < data['MACDh_12_26_9'].iloc[-2]:
            print("(Bearish momentum increasing)")
        elif hist < 0 and hist > data['MACDh_12_26_9'].iloc[-2]:
            print("(Bearish momentum weakening)")
        else:
            print("(Neutral)")

    # ADX Analysis
    if 'ADX_14' in data.columns:
        adx = last_row['ADX_14']
        print(f"ADX: {adx:.2f}", end=" ")

        if adx > 25:
            print("(Strong trend)")
        else:
            print("(Weak trend)")

    # Bollinger Bands Analysis
    if all(col in data.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        close = last_row['close']
        upper_band = last_row['BBU_20_2.0']
        middle_band = last_row['BBM_20_2.0']
        lower_band = last_row['BBL_20_2.0']

        bb_width = (upper_band - lower_band) / middle_band

        print(f"Bollinger Bands: Width = {bb_width:.4f}", end=" ")

        if close > upper_band:
            print("(Price above upper band - potential reversal or strong uptrend)")
        elif close < lower_band:
            print("(Price below lower band - potential reversal or strong downtrend)")
        else:
            distance_to_upper = (upper_band - close) / (upper_band - lower_band) * 100
            print(f"(Price at {distance_to_upper:.1f}% of band range from upper band)")

    # Support/Resistance Analysis
    recent_support = data.iloc[-30:]['support'].max()
    recent_resistance = data.iloc[-30:]['resistance'].max()

    if recent_support > 0:
        print(f"Recent Support Level: ${recent_support:.2f}")
    if recent_resistance > 0:
        print(f"Recent Resistance Level: ${recent_resistance:.2f}")

    # Divergence Analysis
    recent_bullish = data.iloc[-10:]['bullish_divergence'].any() if 'bullish_divergence' in data.columns else False
    recent_bearish = data.iloc[-10:]['bearish_divergence'].any() if 'bearish_divergence' in data.columns else False

    if recent_bullish:
        print("Recent Bullish Divergence Detected!")
    if recent_bearish:
        print("Recent Bearish Divergence Detected!")

    # Multi-timeframe Analysis
    if 'rsi' in data.columns and 'rsi_3d' in data.columns and 'rsi_7d' in data.columns:
        rsi_1d = last_row['rsi']
        rsi_3d = last_row['rsi_3d']
        rsi_7d = last_row['rsi_7d']

        print(f"Multi-timeframe RSI: 1D={rsi_1d:.2f}, 3D={rsi_3d:.2f}, 7D={rsi_7d:.2f}")

        if rsi_1d > rsi_3d > rsi_7d:
            print("RSI Alignment: Short-term bullish momentum exceeding longer-term")
        elif rsi_1d < rsi_3d < rsi_7d:
            print("RSI Alignment: Short-term bearish momentum exceeding longer-term")

    # Final analysis summary
    print("\n=== OVERALL ANALYSIS ===")

    # Determine current trend
    trend = "UNKNOWN"
    if 'ema_8' in data.columns and 'ema_21' in data.columns:
        ema_short = last_row['ema_8']
        ema_medium = last_row['ema_21']
        ema_short_prev = data['ema_8'].iloc[-2]
        ema_medium_prev = data['ema_21'].iloc[-2]

        if ema_short > ema_medium and ema_short_prev > ema_medium_prev:
            trend = "UPTREND"
        elif ema_short < ema_medium and ema_short_prev < ema_medium_prev:
            trend = "DOWNTREND"
        elif ema_short > ema_medium and ema_short_prev < ema_medium_prev:
            trend = "POSSIBLE TREND CHANGE (Bullish crossover)"
        elif ema_short < ema_medium and ema_short_prev > ema_medium_prev:
            trend = "POSSIBLE TREND CHANGE (Bearish crossover)"
        else:
            trend = "SIDEWAYS"

    print(f"Current Market Trend: {trend}")

    # Potential signals
    signals = []

    # RSI signals
    if 'rsi' in data.columns:
        if last_row['rsi'] < 30:
            signals.append("RSI oversold (bullish)")
        elif last_row['rsi'] > 70:
            signals.append("RSI overbought (bearish)")

    # MACD signals
    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
        macd_current = last_row['MACD_12_26_9'] - last_row['MACDs_12_26_9']
        macd_previous = data['MACD_12_26_9'].iloc[-2] - data['MACDs_12_26_9'].iloc[-2]

        if macd_current > 0 and macd_previous < 0:
            signals.append("MACD bullish crossover (bullish)")
        elif macd_current < 0 and macd_previous > 0:
            signals.append("MACD bearish crossover (bearish)")

    # Bollinger Band signals
    if all(col in data.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        if last_row['close'] < last_row['BBL_20_2.0']:
            signals.append("Price below lower Bollinger Band (potential bullish reversal)")
        elif last_row['close'] > last_row['BBU_20_2.0']:
            signals.append("Price above upper Bollinger Band (potential bearish reversal)")

    # Divergence signals
    if recent_bullish:
        signals.append("Bullish divergence detected (bullish)")
    if recent_bearish:
        signals.append("Bearish divergence detected (bearish)")

    if signals:
        print("\nPotential Signals:")
        for signal in signals:
            print(f"- {signal}")
    else:
        print("\nNo clear signals at the moment.")


def main():
    """Main function to execute the indicator demonstration."""
    print("===== TECHNICAL INDICATOR DEMONSTRATION =====")

    # Fetch sample market data
    print("\nFetching market data...")
    data = fetch_sample_data(days=120)
    print(f"Fetched {len(data)} days of data from {data.index[0].date()} to {data.index[-1].date()}")

    # Calculate indicators
    print("\nCalculating technical indicators...")
    result = calculate_all_indicators(data)

    # Display summary of results
    display_results(result)

    # Plot the indicators
    print("\nPlotting indicators...")
    plot_indicators(result)

    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()