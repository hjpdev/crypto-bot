"""
Data Preparation Services for the crypto trading bot.

This module provides functionality for preparing and validating
market data for technical analysis and indicators.
"""

from typing import List, Union, Tuple
import pandas as pd


def ohlcv_to_dataframe(ohlcv_data: List[List[Union[int, float]]]) -> pd.DataFrame:
    """
    Convert OHLCV data from ccxt format to a pandas DataFrame.

    Args:
        ohlcv_data: List of lists containing OHLCV data
            Each inner list should have:
            [timestamp, open, high, low, close, volume]

    Returns:
        pd.DataFrame: DataFrame with OHLCV columns

    Raises:
        ValueError: If data format is invalid
    """
    if not ohlcv_data:
        raise ValueError("OHLCV data cannot be empty")

    df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    return df


def prepare_for_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for technical indicator calculations.

    Args:
        dataframe: DataFrame containing OHLCV data

    Returns:
        pd.DataFrame: Cleaned and prepared DataFrame

    Raises:
        ValueError: If input DataFrame is invalid
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if dataframe.empty:
        raise ValueError("DataFrame cannot be empty")

    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")

    df = dataframe.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        df.sort_index(inplace=True)
    else:
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True)

    # Handle missing values
    # For OHLC, we can use forward fill or a more sophisticated method
    df = df.ffill()
    df.dropna(inplace=True)

    return df


def resample_ohlcv(dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        dataframe: DataFrame containing OHLCV data with DatetimeIndex
        timeframe: Target timeframe (e.g., '1h', '4h', '1d')

    Returns:
        pd.DataFrame: Resampled DataFrame

    Raises:
        ValueError: If input DataFrame is invalid or timeframe format is invalid
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if dataframe.empty:
        raise ValueError("DataFrame cannot be empty")

    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        if "timestamp" in dataframe.columns:
            if not pd.api.types.is_datetime64_any_dtype(dataframe["timestamp"]):
                dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
            df = dataframe.set_index("timestamp")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column")
    else:
        df = dataframe.copy()

    valid_timeframe_units = ["s", "min", "h", "d", "w", "M"]
    if not any(timeframe.endswith(unit) for unit in valid_timeframe_units):
        raise ValueError(
            f"Invalid timeframe format. Must end with one of: {', '.join(valid_timeframe_units)}"
        )

    try:
        resampled = df.resample(timeframe).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )

        resampled.dropna(inplace=True)

        return resampled
    except ValueError as e:
        raise ValueError(
            f"Invalid timeframe format. Must end with one of: {', '.join(valid_timeframe_units)}"
        ) from e


def validate_ohlcv_data(
    data: Union[List[List[Union[int, float]]], pd.DataFrame],
) -> Tuple[bool, str]:
    """
    Validate OHLCV data structure.

    Args:
        data: OHLCV data as a list of lists or DataFrame

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if isinstance(data, pd.DataFrame):
        # Validate DataFrame structure
        required_columns = ["open", "high", "low", "close", "volume"]

        if data.empty:
            return False, "DataFrame is empty"

        # If the index is not a DatetimeIndex, check for timestamp column
        if not isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" not in data.columns:
                return False, "DataFrame must have a DatetimeIndex or a 'timestamp' column"

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"DataFrame missing required columns: {', '.join(missing_columns)}"

        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False, f"Column '{col}' must contain numeric values"

        return True, ""

    elif isinstance(data, list):
        # Validate list of lists structure
        if not data:
            return False, "OHLCV data list is empty"

        # Check each row
        for i, row in enumerate(data):
            if not isinstance(row, list):
                return False, f"Row {i} is not a list"

            if len(row) != 6:
                return (
                    False,
                    f"""
                    Row {i} does not have exactly 6 elements
                    (timestamp, open, high, low, close, volume)
                    """,
                )

            # Check all types in sequence
            for j, (element, element_name) in enumerate(
                zip(row, ["timestamp", "open", "high", "low", "close", "volume"])
            ):
                if not isinstance(element, (int, float)):
                    return (
                        False,
                        f"Row {i}, {element_name} must be a number, got {type(element).__name__}",
                    )

            # Ensure high >= low
            if row[2] < row[3]:
                return False, f"Row {i}: high ({row[2]}) cannot be less than low ({row[3]})"

            # Ensure volume is not negative
            if row[5] < 0:
                return False, f"Row {i}: volume ({row[5]}) cannot be negative"

        return True, ""

    else:
        return False, "Data must be either a pandas DataFrame or a list of lists"
