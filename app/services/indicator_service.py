"""
Technical Indicator Service for the crypto trading bot.

This module provides functionality for calculating various technical indicators
used in trading strategies and analysis.
"""

from typing import Dict, Any
import pandas as pd
import pandas_ta as ta


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
            except Exception as e:
                print(f"Error calculating {indicator_type}: {str(e)}")
                # Continue processing other indicators

        return result_df
