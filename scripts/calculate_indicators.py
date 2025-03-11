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
import argparse
from datetime import datetime, timedelta
import math

# Add the parent directory to sys.path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.services.exchange_service import ExchangeService
from app.services.indicator_service import IndicatorService
from app.services.indicator_utils import normalize_indicator


def fetch_market_data(symbol: str = "BTC/USDT", days: int = 120, timeframe: str = "1d") -> pd.DataFrame:
    """
    Fetch real market data from the exchange.

    Args:
        symbol: Trading pair symbol (e.g., BTC/USDT)
        days: Number of days of historical data to fetch
        timeframe: Timeframe for the candles (e.g., 1d, 4h, 1h)

    Returns:
        DataFrame with OHLCV data
    """
    # Calculate the 'since' timestamp in milliseconds
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    since = int(start_date.timestamp() * 1000)  # Convert to milliseconds
    
    print(f"Fetching {days} days of {timeframe} data for {symbol} since {start_date.strftime('%Y-%m-%d')}")
    
    # Initialize the exchange service (without authentication for public data)
    exchange_service = ExchangeService(
        exchange_id="binance",  # Using Binance as default exchange
        enableRateLimit=True
    )
    
    # Calculate the number of candles needed based on timeframe
    timeframe_to_minutes = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,  # 24 * 60
        "3d": 4320,  # 3 * 24 * 60
        "1w": 10080  # 7 * 24 * 60
    }
    
    # Get minutes in timeframe
    minutes_in_timeframe = timeframe_to_minutes.get(timeframe, 1440)  # Default to 1d if unknown
    
    # Calculate candles needed for the requested days
    minutes_in_period = days * 1440  # days * 24 hours * 60 minutes
    candles_needed = math.ceil(minutes_in_period / minutes_in_timeframe)
    
    # Add buffer for weekends/holidays and exchange limitations
    candles_to_fetch = min(1000, candles_needed * 2)  # Most exchanges limit to 1000 candles
    
    print(f"Calculating approximately {candles_needed} candles needed, fetching {candles_to_fetch}")
    
    # Fetch OHLCV data
    try:
        ohlcv_data = exchange_service.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=candles_to_fetch
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_data, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Check if we have enough data
        if len(df) < 30:  # Most indicators need at least 30 data points
            print(f"Warning: Only {len(df)} candles were fetched, which is not enough for reliable indicator calculation")
            if len(df) < 2:
                print("Error: Not enough data points. Falling back to sample data...")
                return fetch_sample_data(days)
        
        print(f"Successfully fetched {len(df)} candles")
        return df
        
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        print("Falling back to sample data...")
        return fetch_sample_data(days)


def fetch_sample_data(days: int = 100) -> pd.DataFrame:
    """
    Create sample market data for demonstration purposes.
    This is used as a fallback when real data cannot be fetched.

    Args:
        days: Number of days of data to generate

    Returns:
        DataFrame with OHLCV data
    """
    print("Generating sample market data...")

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
    # Check if we have enough data for calculations
    min_data_points = 35  # Need at least this many for MACD and other indicators
    if len(data) < min_data_points:
        print(f"Warning: Only {len(data)} data points available. At least {min_data_points} recommended for reliable calculations.")
    
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

    # Check for MACD specifically as it needs the most data
    if len(data) < 35:
        print(f"Skipping MACD calculation: Not enough data points for MACD calculation. Need at least 35 periods.")
        # Remove MACD from config if not enough data
        if 'macd' in indicators_config:
            del indicators_config['macd']

    # Calculate all indicators using batch_calculate
    try:
        result = IndicatorService.batch_calculate(data, indicators_config)
    except Exception as e:
        print(f"Error in batch calculation: {str(e)}")
        # Create empty result dataframe
        result = pd.DataFrame(index=data.index)
        # Add price data at minimum
        result['close'] = data['close']
        result['open'] = data['open']
        result['high'] = data['high']
        result['low'] = data['low']
        result['volume'] = data['volume']

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
    print("\n=== SUMMARY STATISTICS ===")
    
    # Current price
    try:
        current_price = data['close'].iloc[-1]
        print(f"Current Price: ${current_price:.2f}")
    except (KeyError, IndexError):
        print("Current Price: Not available")
    
    # RSI
    try:
        rsi_value = data.get('rsi', pd.Series()).iloc[-1]
        if pd.notnull(rsi_value):
            print(f"RSI: {rsi_value:.2f}", end=" ")
            if rsi_value < 30:
                print("(Oversold)")
            elif rsi_value > 70:
                print("(Overbought)")
            else:
                print("(Neutral)")
        else:
            print("RSI: Not available")
    except (KeyError, IndexError, AttributeError):
        print("RSI: Not available")
    
    # MACD
    try:
        macd_value = data.get('macd', pd.Series()).iloc[-1]
        signal_value = data.get('macd_signal', pd.Series()).iloc[-1]
        histogram_value = data.get('macd_histogram', pd.Series()).iloc[-1]
        
        if pd.notnull(macd_value) and pd.notnull(signal_value) and pd.notnull(histogram_value):
            print(f"MACD: {macd_value:.4f}, Signal: {signal_value:.4f}, Histogram: {histogram_value:.4f}", end=" ")
            
            if histogram_value > 0 and histogram_value > data.get('macd_histogram', pd.Series()).iloc[-2]:
                print("(Bullish momentum strengthening)")
            elif histogram_value > 0 and histogram_value < data.get('macd_histogram', pd.Series()).iloc[-2]:
                print("(Bullish momentum weakening)")
            elif histogram_value < 0 and histogram_value < data.get('macd_histogram', pd.Series()).iloc[-2]:
                print("(Bearish momentum strengthening)")
            elif histogram_value < 0 and histogram_value > data.get('macd_histogram', pd.Series()).iloc[-2]:
                print("(Bearish momentum weakening)")
            else:
                print("(Neutral)")
        else:
            print("MACD: Not available")
    except (KeyError, IndexError, AttributeError):
        print("MACD: Not available")
    
    # ADX
    try:
        adx_value = data.get('adx', pd.Series()).iloc[-1]
        if pd.notnull(adx_value):
            print(f"ADX: {adx_value:.2f}", end=" ")
            if adx_value > 25:
                print("(Strong trend)")
            elif adx_value > 20:
                print("(Trending)")
            else:
                print("(No trend)")
        else:
            print("ADX: Not available")
    except (KeyError, IndexError, AttributeError):
        print("ADX: Not available")
    
    # Bollinger Bands
    try:
        bb_upper = data.get('bb_upper', pd.Series()).iloc[-1]
        bb_lower = data.get('bb_lower', pd.Series()).iloc[-1]
        close = data['close'].iloc[-1]
        
        if pd.notnull(bb_upper) and pd.notnull(bb_lower):
            bb_width = (bb_upper - bb_lower) / data.get('bb_middle', pd.Series()).iloc[-1]
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            print(f"Bollinger Bands: Width = {bb_width:.4f}", end=" ")
            print(f"(Price at {bb_position*100:.1f}% of band range from upper band)")
        else:
            print("Bollinger Bands: Not available")
    except (KeyError, IndexError, AttributeError, ZeroDivisionError):
        print("Bollinger Bands: Not available")
    
    # Support/Resistance
    try:
        if 'resistance_level' in data.columns and pd.notnull(data['resistance_level'].iloc[-1]):
            resistance = data['resistance_level'].iloc[-1]
            print(f"Recent Resistance Level: ${resistance:.2f}")
        else:
            print("Resistance Level: Not available")
            
        if 'support_level' in data.columns and pd.notnull(data['support_level'].iloc[-1]):
            support = data['support_level'].iloc[-1]
            print(f"Recent Support Level: ${support:.2f}")
        else:
            print("Support Level: Not available")
    except (KeyError, IndexError, AttributeError):
        print("Support/Resistance Levels: Not available")
    
    # Multi-timeframe RSI
    try:
        rsi_1d = data.get('rsi', pd.Series()).iloc[-1] if 'rsi' in data.columns else None
        rsi_3d = data.get('rsi_3d', pd.Series()).iloc[-1] if 'rsi_3d' in data.columns else None
        rsi_7d = data.get('rsi_7d', pd.Series()).iloc[-1] if 'rsi_7d' in data.columns else None
        
        if any(pd.notnull(x) for x in [rsi_1d, rsi_3d, rsi_7d]):
            print("Multi-timeframe RSI:", end=" ")
            rsi_parts = []
            if pd.notnull(rsi_1d):
                rsi_parts.append(f"1D={rsi_1d:.2f}")
            if pd.notnull(rsi_3d):
                rsi_parts.append(f"3D={rsi_3d:.2f}")
            if pd.notnull(rsi_7d):
                rsi_parts.append(f"7D={rsi_7d:.2f}")
            print(", ".join(rsi_parts))
        else:
            print("Multi-timeframe RSI: Not available")
    except (KeyError, IndexError, AttributeError):
        print("Multi-timeframe RSI: Not available")
    
    # Overall trend analysis
    print("\n=== OVERALL ANALYSIS ===")
    try:
        close_prices = data['close']
        last_10_days = close_prices.iloc[-10:]
        
        # Simple trend detection
        if len(last_10_days) >= 5:
            trend_direction = "UNDEFINED"
            
            # Use SMA or EMA if available for trend detection
            if 'ema_21' in data.columns and pd.notnull(data['ema_21'].iloc[-1]):
                if close_prices.iloc[-1] > data['ema_21'].iloc[-1] and close_prices.iloc[-5] > data['ema_21'].iloc[-5]:
                    trend_direction = "UPTREND"
                elif close_prices.iloc[-1] < data['ema_21'].iloc[-1] and close_prices.iloc[-5] < data['ema_21'].iloc[-5]:
                    trend_direction = "DOWNTREND"
                elif close_prices.iloc[-1] > data['ema_21'].iloc[-1] and close_prices.iloc[-5] < data['ema_21'].iloc[-5]:
                    trend_direction = "REVERSAL UP"
                elif close_prices.iloc[-1] < data['ema_21'].iloc[-1] and close_prices.iloc[-5] > data['ema_21'].iloc[-5]:
                    trend_direction = "REVERSAL DOWN"
            else:
                # Simple price comparison if no moving averages
                if close_prices.iloc[-1] > close_prices.iloc[-5]:
                    trend_direction = "UPTREND"
                elif close_prices.iloc[-1] < close_prices.iloc[-5]:
                    trend_direction = "DOWNTREND"
                
            print(f"Current Market Trend: {trend_direction}")
        else:
            print("Current Market Trend: Not enough data")
    except (KeyError, IndexError, AttributeError):
        print("Current Market Trend: Could not determine")
    
    # Potential signals
    signals = []

    # RSI signals
    try:
        if 'rsi' in data.columns and pd.notnull(data['rsi'].iloc[-1]):
            if data['rsi'].iloc[-1] < 30:
                signals.append("RSI oversold (bullish)")
            elif data['rsi'].iloc[-1] > 70:
                signals.append("RSI overbought (bearish)")
    except (KeyError, IndexError, AttributeError):
        pass

    # MACD signals
    try:
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd_current = data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]
            macd_previous = data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]

            if macd_current > 0 and macd_previous < 0:
                signals.append("MACD bullish crossover (bullish)")
            elif macd_current < 0 and macd_previous > 0:
                signals.append("MACD bearish crossover (bearish)")
    except (KeyError, IndexError, AttributeError):
        pass

    # Bollinger Band signals
    try:
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            if data['close'].iloc[-1] < data['bb_lower'].iloc[-1]:
                signals.append("Price below lower Bollinger Band (potential bullish reversal)")
            elif data['close'].iloc[-1] > data['bb_upper'].iloc[-1]:
                signals.append("Price above upper Bollinger Band (potential bearish reversal)")
    except (KeyError, IndexError, AttributeError):
        pass

    # Divergence signals
    try:
        if 'bullish_divergence' in data.columns and data['bullish_divergence'].iloc[-1]:
            signals.append("Bullish divergence detected (bullish)")
        if 'bearish_divergence' in data.columns and data['bearish_divergence'].iloc[-1]:
            signals.append("Bearish divergence detected (bearish)")
    except (KeyError, IndexError, AttributeError):
        pass

    if signals:
        print("\nPotential Signals:")
        for signal in signals:
            print(f"- {signal}")
    else:
        print("\nNo clear signals at the moment.")


def main():
    """Main function to execute the indicator demonstration."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate technical indicators for a given symbol")
    parser.add_argument("--symbol", type=str, default="BTC/USDT",
                        help="Trading pair symbol (e.g., BTC/USDT)")
    parser.add_argument("--days", type=int, default=120,
                        help="Number of days of historical data to fetch")
    parser.add_argument("--timeframe", type=str, default="1d",
                        help="Timeframe for candles (e.g., 1d, 4h, 1h)")
    args = parser.parse_args()

    print("===== TECHNICAL INDICATOR DEMONSTRATION =====")
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}, Days: {args.days}")

    # Fetch market data
    print("\nFetching market data...")
    data = fetch_market_data(symbol=args.symbol, days=args.days, timeframe=args.timeframe)
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