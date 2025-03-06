"""
Technical Indicator Service for the crypto trading bot.

This module provides functionality for calculating various technical indicators
used in trading strategies and analysis.
"""

from typing import Dict, Any, Callable
import pandas as pd
import pandas_ta as ta
import numpy as np


class IndicatorService:
    """
    Service for calculating technical indicators on market data.

    This class provides methods to calculate common technical indicators
    used in trading strategies, such as RSI, MACD, EMAs, and more.
    It can handle both pandas DataFrames and arrays/lists as input.
    """

    @staticmethod
    def validate_dataframe(dataframe: pd.DataFrame, required_column: str = "close") -> bool:
        """Validates that the input dataframe contains the required columns."""
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty")

        if required_column not in dataframe.columns:
            raise ValueError(f"DataFrame must contain a '{required_column}' column")

        return True

    @classmethod
    def calculate_rsi(
        cls, dataframe: pd.DataFrame, period: int = 14, column: str = "close"
    ) -> pd.DataFrame:
        """Calculate the Relative Strength Index (RSI) for the given dataframe."""
        cls.validate_dataframe(dataframe, column)

        if period < 2:
            raise ValueError("Period must be at least 2")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()
        df["rsi"] = ta.rsi(df[column], length=period)

        return df

    @classmethod
    def calculate_macd(
        cls,
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing price data
            fast: Fast period for MACD calculation (default: 12)
            slow: Slow period for MACD calculation (default: 26)
            signal: Signal period for MACD calculation (default: 9)
            column: The column to use for calculation (default: 'close')

        Returns:
            DataFrame with the original data and additional MACD columns

        Raises:
            ValueError: If input data is invalid
        """
        cls.validate_dataframe(dataframe, column)

        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
        if signal < 1:
            raise ValueError("Signal period must be at least 1")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Check if we have enough data for MACD calculation
        min_periods = slow + signal
        if len(df) < min_periods:
            raise ValueError(
                f"Not enough data points for MACD calculation. Need at least {min_periods} periods."
            )

        # Calculate MACD using pandas-ta
        macd_result = ta.macd(df[column], fast=fast, slow=slow, signal=signal)

        # Merge the MACD results with the original dataframe
        if macd_result is not None:
            df = pd.concat([df, macd_result], axis=1)

        return df

    @classmethod
    def calculate_ema(
        cls, dataframe: pd.DataFrame, period: int, column: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate the Exponential Moving Average (EMA) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing price data
            period: The period over which to calculate EMA
            column: The column to use for calculation (default: 'close')

        Returns:
            DataFrame with the original data and an additional 'ema_{period}' column

        Raises:
            ValueError: If input data is invalid
        """
        cls.validate_dataframe(dataframe, column)

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()
        col_name = f"ema_{period}"
        df[col_name] = ta.ema(df[column], length=period)

        return df

    @classmethod
    def calculate_sma(
        cls, dataframe: pd.DataFrame, period: int, column: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate the Simple Moving Average (SMA) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing price data
            period: The period over which to calculate SMA
            column: The column to use for calculation (default: 'close')

        Returns:
            DataFrame with the original data and an additional 'sma_{period}' column

        Raises:
            ValueError: If input data is invalid
        """
        cls.validate_dataframe(dataframe, column)

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()
        col_name = f"sma_{period}"
        df[col_name] = ta.sma(df[column], length=period)

        return df

    @classmethod
    def calculate_roc(
        cls, dataframe: pd.DataFrame, period: int, column: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate the Rate of Change (ROC) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing price data
            period: The period over which to calculate ROC
            column: The column to use for calculation (default: 'close')

        Returns:
            DataFrame with the original data and an additional 'roc_{period}' column

        Raises:
            ValueError: If input data is invalid
        """
        cls.validate_dataframe(dataframe, column)

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()
        col_name = f"roc_{period}"
        df[col_name] = ta.roc(df[column], length=period)

        return df

    @classmethod
    def calculate_bollinger_bands(
        cls, dataframe: pd.DataFrame, period: int = 20, std_dev: int = 2, column: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing price data
            period: The period over which to calculate the moving average (default: 20)
            std_dev: Number of standard deviations for the bands (default: 2)
            column: The column to use for calculation (default: 'close')

        Returns:
            DataFrame with the original data and additional Bollinger Bands columns:
            - 'bb_upper': Upper Bollinger Band
            - 'bb_middle': Middle Band (SMA)
            - 'bb_lower': Lower Bollinger Band
            - 'bb_width': Band width ((upper - lower) / middle)

        Raises:
            ValueError: If input data is invalid
        """
        cls.validate_dataframe(dataframe, column)

        if period < 2:
            raise ValueError("Period must be at least 2")
        if std_dev < 0:
            raise ValueError("Standard deviation must be non-negative")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Calculate Bollinger Bands using pandas-ta
        bb_result = ta.bbands(df[column], length=period, std=std_dev)

        # Merge the results with the original dataframe
        if bb_result is not None:
            df = pd.concat([df, bb_result], axis=1)

        return df

    @classmethod
    def calculate_atr(cls, dataframe: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing OHLC price data
            period: The period over which to calculate ATR (default: 14)

        Returns:
            DataFrame with the original data and an additional 'ATR_{period}' column

        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            cls.validate_dataframe(dataframe, col)

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Calculate ATR using pandas-ta
        atr_value = ta.atr(df["high"], df["low"], df["close"], length=period)

        # Add with the appropriate column name
        df[f"ATR_{period}"] = atr_value

        return df

    @classmethod
    def calculate_adx(cls, dataframe: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing OHLC price data
            period: The period over which to calculate ADX (default: 14)

        Returns:
            DataFrame with the original data and additional ADX columns:
            - 'ADX_{period}': Average Directional Index
            - 'DMP_{period}': Plus Directional Movement (DI+)
            - 'DMN_{period}': Minus Directional Movement (DI-)

        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            cls.validate_dataframe(dataframe, col)

        if period < 2:
            raise ValueError("Period must be at least 2")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Calculate ADX using pandas-ta
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=period)

        # Merge the results with the original dataframe
        if adx_result is not None:
            # Rename columns to match expected naming pattern
            adx_result.columns = [f"ADX_{period}", f"DMP_{period}", f"DMN_{period}"]
            df = pd.concat([df, adx_result], axis=1)

        return df

    @classmethod
    def calculate_obv(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing price and volume data

        Returns:
            DataFrame with the original data and an additional 'obv' column

        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        required_columns = ["close", "volume"]
        for col in required_columns:
            cls.validate_dataframe(dataframe, col)

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Calculate OBV using pandas-ta
        df["obv"] = ta.obv(df["close"], df["volume"])

        return df

    @classmethod
    def calculate_vwap(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume-Weighted Average Price (VWAP) for the given dataframe.

        Args:
            dataframe: Pandas DataFrame containing OHLCV price data

        Returns:
            DataFrame with the original data and an additional 'vwap' column

        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        required_columns = ["high", "low", "close", "volume"]
        for col in required_columns:
            cls.validate_dataframe(dataframe, col)

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Calculate typical price: (high + low + close) / 3
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate VWAP
        df["tp_volume"] = df["typical_price"] * df["volume"]
        df["cumulative_tp_volume"] = df["tp_volume"].cumsum()
        df["cumulative_volume"] = df["volume"].cumsum()

        # Avoid division by zero
        df["vwap"] = np.where(
            df["cumulative_volume"] > 0,
            df["cumulative_tp_volume"] / df["cumulative_volume"],
            df["typical_price"],
        )

        # Remove temporary columns
        df.drop(
            ["typical_price", "tp_volume", "cumulative_tp_volume", "cumulative_volume"],
            axis=1,
            inplace=True,
        )

        return df

    @classmethod
    def calculate_support_resistance(
        cls, dataframe: pd.DataFrame, lookback: int = 14
    ) -> pd.DataFrame:
        """
        Identify support and resistance levels using recent swing highs and lows.

        Args:
            dataframe: Pandas DataFrame containing price data
            lookback: Number of periods to look back for identifying swings (default: 14)

        Returns:
            DataFrame with the original data and additional columns:
            - 'support': Identified support levels (non-zero where support is identified)
            - 'resistance': Identified resistance levels (non-zero where resistance is identified)

        Raises:
            ValueError: If input data is invalid
        """
        from app.services.indicator_utils import find_swing_highs, find_swing_lows

        cls.validate_dataframe(dataframe, "close")

        if lookback < 3:
            raise ValueError("Lookback period must be at least 3")

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Find swing highs and lows using the utility functions
        swing_highs = find_swing_highs(df["high"], window=lookback)
        swing_lows = find_swing_lows(df["low"], window=lookback)

        # Create support and resistance columns
        df["resistance"] = 0.0
        df["support"] = 0.0

        # Convert integer positions to actual index values
        if len(swing_highs) > 0:
            # Get the actual index values corresponding to the swing high positions
            swing_high_indices = df.index[swing_highs]
            # Mark swing highs as resistance
            df.loc[swing_high_indices, "resistance"] = df.loc[swing_high_indices, "high"]

        if len(swing_lows) > 0:
            # Get the actual index values corresponding to the swing low positions
            swing_low_indices = df.index[swing_lows]
            # Mark swing lows as support
            df.loc[swing_low_indices, "support"] = df.loc[swing_low_indices, "low"]

        return df

    @classmethod
    def calculate_multi_timeframe(
        cls, dataframes_dict: Dict[str, pd.DataFrame], indicator_func: Callable, **params
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate an indicator across multiple timeframes.

        Args:
            dataframes_dict: Dictionary mapping timeframe names to DataFrames
            indicator_func: Function reference to the indicator calculation method
            **params: Additional parameters to pass to the indicator function

        Returns:
            Dictionary with the same keys as dataframes_dict, but with
            DataFrames containing the calculated indicator

        Raises:
            ValueError: If input data is invalid
        """
        if not dataframes_dict:
            raise ValueError("Dataframes dictionary cannot be empty")

        result = {}

        for timeframe, df in dataframes_dict.items():
            try:
                # Apply the indicator function to each dataframe
                result[timeframe] = indicator_func(df, **params)
            except Exception as e:
                print(f"Error calculating indicator for timeframe {timeframe}: {str(e)}")
                # Skip this timeframe but continue processing others

        return result

    @classmethod
    def detect_divergence(
        cls, price_data: pd.Series, indicator_data: pd.Series, window: int = 10
    ) -> pd.DataFrame:
        """
        Detect bullish and bearish divergence between price and indicator.

        Args:
            price_data: Pandas Series containing price data
            indicator_data: Pandas Series containing indicator values
            window: Lookback window for divergence detection (default: 10)

        Returns:
            DataFrame with columns:
            - 'price': Original price data
            - 'indicator': Original indicator data
            - 'bullish_divergence': Boolean True where bullish divergence is detected
            - 'bearish_divergence': Boolean True where bearish divergence is detected

        Raises:
            ValueError: If input data is invalid
        """
        from app.services.indicator_utils import find_swing_highs, find_swing_lows

        if not isinstance(price_data, pd.Series) or not isinstance(indicator_data, pd.Series):
            raise ValueError("Price and indicator data must be pandas Series")

        if len(price_data) != len(indicator_data):
            raise ValueError("Price and indicator data must have the same length")

        if window < 3:
            raise ValueError("Window must be at least 3")

        # Create a resulting dataframe
        result = pd.DataFrame(
            {
                "price": price_data,
                "indicator": indicator_data,
                "bullish_divergence": False,
                "bearish_divergence": False,
            }
        )

        # Find swing highs and lows in price and indicator
        price_swing_highs = find_swing_highs(price_data, window=window)
        price_swing_lows = find_swing_lows(price_data, window=window)
        indicator_swing_highs = find_swing_highs(indicator_data, window=window)
        indicator_swing_lows = find_swing_lows(indicator_data, window=window)

        # Detect bearish divergence: price making higher highs, indicator making lower highs
        for i in range(1, len(price_data)):
            if i in price_swing_highs and i - window > 0:
                # Find the previous swing high within the window
                prev_highs = [j for j in price_swing_highs if j < i and j >= i - window]

                if prev_highs and price_data[i] > price_data[prev_highs[-1]]:
                    # Check if indicator made lower highs in the same period
                    indicator_highs = [
                        j for j in indicator_swing_highs if j >= prev_highs[-1] and j <= i
                    ]

                    if (
                        len(indicator_highs) >= 2
                        and indicator_data[indicator_highs[-1]] < indicator_data[indicator_highs[0]]
                    ):
                        result.loc[i, "bearish_divergence"] = True

        # Detect bullish divergence: price making lower lows, indicator making higher lows
        for i in range(1, len(price_data)):
            if i in price_swing_lows and i - window > 0:
                # Find the previous swing low within the window
                prev_lows = [j for j in price_swing_lows if j < i and j >= i - window]

                if prev_lows and price_data[i] < price_data[prev_lows[-1]]:
                    # Check if indicator made higher lows in the same period
                    indicator_lows = [
                        j for j in indicator_swing_lows if j >= prev_lows[-1] and j <= i
                    ]

                    if (
                        len(indicator_lows) >= 2
                        and indicator_data[indicator_lows[-1]] > indicator_data[indicator_lows[0]]
                    ):
                        result.loc[i, "bullish_divergence"] = True

        return result

    @classmethod
    def batch_calculate(
        cls, dataframe: pd.DataFrame, indicators_config: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calculate multiple indicators in a single batch operation.

        Args:
            dataframe: Pandas DataFrame containing price data
            indicators_config: Dictionary of indicators to calculate with their parameters
                Example:
                {
                    'rsi': {'period': 14, 'column': 'close'},
                    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                    'ema': [{'period': 9}, {'period': 21}]
                }

        Returns:
            DataFrame with the original data and additional indicator columns

        Raises:
            ValueError: If input data or configuration is invalid
        """
        cls.validate_dataframe(dataframe)

        if not indicators_config:
            raise ValueError("Indicators configuration cannot be empty")

        # Create a copy to avoid modifying the original
        result_df = dataframe.copy()

        # Process each indicator type
        for indicator_type, config in indicators_config.items():
            try:
                if indicator_type == "rsi":
                    # Handle single RSI calculation
                    params = config if isinstance(config, dict) else {}
                    temp_df = cls.calculate_rsi(result_df, **params)
                    if "rsi" in temp_df.columns:
                        result_df["rsi"] = temp_df["rsi"]

                elif indicator_type == "macd":
                    # Handle MACD calculation
                    params = config if isinstance(config, dict) else {}
                    try:
                        temp_df = cls.calculate_macd(result_df, **params)
                        # Get only the MACD columns
                        macd_cols = [col for col in temp_df.columns if col.startswith("MACD")]
                        if macd_cols:
                            for col in macd_cols:
                                result_df[col] = temp_df[col]
                    except ValueError as e:
                        # Skip MACD if not enough data
                        print(f"Skipping MACD calculation: {str(e)}")

                elif indicator_type == "ema":
                    # Handle one or multiple EMA calculations
                    if isinstance(config, list):
                        for ema_config in config:
                            params = ema_config if isinstance(ema_config, dict) else {}
                            period = params.get("period")
                            if not period:
                                continue
                            temp_df = cls.calculate_ema(
                                result_df, period=period, column=params.get("column", "close")
                            )
                            col_name = f"ema_{period}"
                            result_df[col_name] = temp_df[col_name]
                    else:
                        params = config if isinstance(config, dict) else {}
                        period = params.get("period")
                        if period:
                            temp_df = cls.calculate_ema(
                                result_df, period=period, column=params.get("column", "close")
                            )
                            col_name = f"ema_{period}"
                            result_df[col_name] = temp_df[col_name]

                elif indicator_type == "sma":
                    # Handle one or multiple SMA calculations
                    if isinstance(config, list):
                        for sma_config in config:
                            params = sma_config if isinstance(sma_config, dict) else {}
                            period = params.get("period")
                            if not period:
                                continue
                            temp_df = cls.calculate_sma(
                                result_df, period=period, column=params.get("column", "close")
                            )
                            col_name = f"sma_{period}"
                            result_df[col_name] = temp_df[col_name]
                    else:
                        params = config if isinstance(config, dict) else {}
                        period = params.get("period")
                        if period:
                            temp_df = cls.calculate_sma(
                                result_df, period=period, column=params.get("column", "close")
                            )
                            col_name = f"sma_{period}"
                            result_df[col_name] = temp_df[col_name]

                elif indicator_type == "roc":
                    # Handle one or multiple ROC calculations
                    if isinstance(config, list):
                        for roc_config in config:
                            params = roc_config if isinstance(roc_config, dict) else {}
                            period = params.get("period")
                            if not period:
                                continue
                            temp_df = cls.calculate_roc(
                                result_df, period=period, column=params.get("column", "close")
                            )
                            col_name = f"roc_{period}"
                            result_df[col_name] = temp_df[col_name]
                    else:
                        params = config if isinstance(config, dict) else {}
                        period = params.get("period")
                        if period:
                            temp_df = cls.calculate_roc(
                                result_df, period=period, column=params.get("column", "close")
                            )
                            col_name = f"roc_{period}"
                            result_df[col_name] = temp_df[col_name]

                elif indicator_type == "bollinger_bands":
                    # Handle Bollinger Bands calculation
                    params = config if isinstance(config, dict) else {}
                    temp_df = cls.calculate_bollinger_bands(result_df, **params)
                    # Get the Bollinger Bands columns
                    bb_cols = [
                        col
                        for col in temp_df.columns
                        if col.startswith("BBL")
                        or col.startswith("BBM")
                        or col.startswith("BBU")
                        or col.startswith("BBB")
                    ]
                    if bb_cols:
                        for col in bb_cols:
                            result_df[col] = temp_df[col]

                elif indicator_type == "atr":
                    # Handle ATR calculation
                    params = config if isinstance(config, dict) else {}
                    temp_df = cls.calculate_atr(result_df, **params)
                    if f"ATR_{params.get('period')}" in temp_df.columns:
                        result_df[f"ATR_{params.get('period')}"] = temp_df[
                            f"ATR_{params.get('period')}"
                        ]

                elif indicator_type == "adx":
                    # Handle ADX calculation
                    params = config if isinstance(config, dict) else {}
                    temp_df = cls.calculate_adx(result_df, **params)
                    # Get the ADX columns
                    adx_cols = [
                        col
                        for col in temp_df.columns
                        if col.startswith("ADX") or col.startswith("DMP") or col.startswith("DMN")
                    ]
                    if adx_cols:
                        for col in adx_cols:
                            result_df[col] = temp_df[col]

                elif indicator_type == "obv":
                    # Handle OBV calculation
                    temp_df = cls.calculate_obv(result_df)
                    if "obv" in temp_df.columns:
                        result_df["obv"] = temp_df["obv"]

                elif indicator_type == "vwap":
                    # Handle VWAP calculation
                    temp_df = cls.calculate_vwap(result_df)
                    if "vwap" in temp_df.columns:
                        result_df["vwap"] = temp_df["vwap"]

                elif indicator_type == "support_resistance":
                    # Handle support/resistance calculation
                    params = config if isinstance(config, dict) else {}
                    temp_df = cls.calculate_support_resistance(result_df, **params)
                    if "support" in temp_df.columns and "resistance" in temp_df.columns:
                        result_df["support"] = temp_df["support"]
                        result_df["resistance"] = temp_df["resistance"]

            except Exception as e:
                print(f"Error calculating {indicator_type}: {str(e)}")
                # Continue processing other indicators

        return result_df
