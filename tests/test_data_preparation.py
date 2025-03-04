"""
Unit tests for the data_preparation module.
"""
import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame
from datetime import datetime
from app.services.data_preparation import (
    ohlcv_to_dataframe,
    prepare_for_indicators,
    resample_ohlcv,
    validate_ohlcv_data
)


@pytest.fixture
def sample_ohlcv_list_data() -> list:
    """
    Create a sample OHLCV data in list format for testing.
    """
    # Create a list of OHLCV data with 20 entries
    # Format: [timestamp, open, high, low, close, volume]
    current_time = int(datetime(2023, 1, 1).timestamp() * 1000)
    interval = 3600 * 1000  # 1 hour in milliseconds

    # Create price data with a simple pattern
    base_price = 100
    data = []

    for i in range(20):
        timestamp = current_time + (i * interval)
        close = base_price + (i % 10)
        open_price = base_price + ((i - 1) % 10 if i > 0 else 0)
        high = max(open_price, close) + 1
        low = min(open_price, close) - 1
        volume = 1000 + (i * 100)

        data.append([timestamp, open_price, high, low, close, volume])

    return data


@pytest.fixture
def sample_ohlcv_dataframe() -> DataFrame:
    """
    Create a sample OHLCV DataFrame for testing.
    """
    # Create a sample DataFrame with hourly data for 3 days
    dates = pd.date_range(start='2023-01-01', periods=72, freq='h')

    # Create price data with a simple pattern
    close_prices = []
    for i in range(72):
        # Create a price wave pattern
        price = 100 + 10 * np.sin(i * np.pi / 12)
        close_prices.append(price)

    # Generate other OHLCV data based on close prices
    data = {
        'open': [close_prices[i-1] if i > 0 else close_prices[0] for i in range(72)],
        'high': [p + 1 for p in close_prices],
        'low': [p - 1 for p in close_prices],
        'close': close_prices,
        'volume': [1000 + i * 10 for i in range(72)]
    }

    df = pd.DataFrame(data, index=dates)
    return df


class TestDataPreparation:
    """Test cases for the data_preparation module."""

    def test_ohlcv_to_dataframe(self, sample_ohlcv_list_data):
        """Test conversion from OHLCV list data to DataFrame."""
        # Convert list data to DataFrame
        df = ohlcv_to_dataframe(sample_ohlcv_list_data)

        # Check that the DataFrame has the expected shape and columns
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == len(sample_ohlcv_list_data)
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']

        # Check that the index is a DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check that the values match the original data
        for i, row in enumerate(sample_ohlcv_list_data):
            assert df.iloc[i]['open'] == row[1]
            assert df.iloc[i]['high'] == row[2]
            assert df.iloc[i]['low'] == row[3]
            assert df.iloc[i]['close'] == row[4]
            assert df.iloc[i]['volume'] == row[5]

        # Test with empty data
        with pytest.raises(ValueError, match="OHLCV data cannot be empty"):
            ohlcv_to_dataframe([])

    def test_prepare_for_indicators(self, sample_ohlcv_dataframe):
        """Test preparation of DataFrame for indicators."""
        # Create a copy with some missing values
        df_with_nans = sample_ohlcv_dataframe.copy()
        df_with_nans.iloc[5:10, 0:4] = np.nan

        # Prepare the DataFrame
        prepared_df = prepare_for_indicators(df_with_nans)

        # Check that there are no NaN values
        assert not prepared_df.isna().any().any()

        # Original DataFrame should be unchanged
        assert df_with_nans.isna().any().any()

        # Test with invalid inputs
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            prepare_for_indicators([1, 2, 3])

        # Test with empty DataFrame
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            prepare_for_indicators(pd.DataFrame())

        # Test with missing required columns
        df_missing_col = sample_ohlcv_dataframe.drop('close', axis=1)
        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            prepare_for_indicators(df_missing_col)

    def test_resample_ohlcv(self, sample_ohlcv_dataframe):
        """Test resampling of OHLCV data to different timeframes."""
        # Original data is hourly, resample to 4h
        resampled_4h = resample_ohlcv(sample_ohlcv_dataframe, '4h')

        # Check that the resampled DataFrame has the expected shape
        assert resampled_4h.shape[0] == sample_ohlcv_dataframe.shape[0] // 4

        # Check that the columns are preserved
        assert list(resampled_4h.columns) == list(sample_ohlcv_dataframe.columns)

        # Resample to daily
        resampled_1d = resample_ohlcv(sample_ohlcv_dataframe, '1d')
        assert resampled_1d.shape[0] == sample_ohlcv_dataframe.shape[0] // 24

        # Test with invalid timeframe format
        with pytest.raises(ValueError, match="Invalid timeframe format"):
            resample_ohlcv(sample_ohlcv_dataframe, 'invalid')

        # Test with DataFrame without DatetimeIndex and no timestamp column
        df_no_datetime_index = sample_ohlcv_dataframe.reset_index()
        df_no_timestamp = df_no_datetime_index.rename(columns={'index': 'date'})
        with pytest.raises(
            ValueError,
            match="DataFrame must have a DatetimeIndex or a 'timestamp' column"
        ):
            resample_ohlcv(df_no_timestamp, '4h')

        # Test with DataFrame with timestamp column
        df_with_timestamp = sample_ohlcv_dataframe.reset_index().rename(
            columns={'index': 'timestamp'}
        )
        resampled = resample_ohlcv(df_with_timestamp, '4h')
        assert resampled.shape[0] == sample_ohlcv_dataframe.shape[0] // 4

    def test_validate_ohlcv_data_list(self):
        """Test validation of OHLCV list data."""
        # Create a valid dataset for testing
        valid_data = [
            [1672534800000, 100.0, 102.0, 99.0, 101.0, 1100.0],
            [1672538400000, 101.0, 103.0, 100.0, 102.0, 1200.0],
            [1672542000000, 102.0, 104.0, 101.0, 103.0, 1300.0]
        ]

        # Valid data should return True
        is_valid, _ = validate_ohlcv_data(valid_data)
        assert is_valid is True

        # Test with empty list
        is_valid, error_msg = validate_ohlcv_data([])
        assert is_valid is False
        assert "empty" in error_msg

        # Test with invalid row format (missing volume)
        invalid_row_data = [
            [1672534800000, 100.0, 102.0, 99.0, 101.0]  # Missing volume
        ]
        is_valid, error_msg = validate_ohlcv_data(invalid_row_data)
        assert is_valid is False
        assert "exactly 6 elements" in error_msg

        # Test with non-numeric open
        non_numeric_open = [
            [1672534800000, "invalid", 102.0, 99.0, 101.0, 1100.0]
        ]
        is_valid, error_msg = validate_ohlcv_data(non_numeric_open)
        assert is_valid is False
        assert "open must be a number" in error_msg

        # Test with non-numeric timestamp
        non_numeric_timestamp = [
            ["invalid", 100.0, 102.0, 99.0, 101.0, 1100.0]
        ]
        is_valid, error_msg = validate_ohlcv_data(non_numeric_timestamp)
        assert is_valid is False
        assert "timestamp must be a number" in error_msg

        # Test with high < low
        invalid_high_low = [
            [1672534800000, 100.0, 95.0, 98.0, 101.0, 1100.0]  # High < Low
        ]
        is_valid, error_msg = validate_ohlcv_data(invalid_high_low)
        assert is_valid is False
        assert "high" in error_msg and "low" in error_msg

        # Test with negative volume
        negative_volume = [
            [1672534800000, 100.0, 102.0, 99.0, 101.0, -1100.0]  # Negative volume
        ]
        is_valid, error_msg = validate_ohlcv_data(negative_volume)
        assert is_valid is False
        assert "volume" in error_msg and "negative" in error_msg

    def test_validate_ohlcv_data_dataframe(self, sample_ohlcv_dataframe):
        """Test validation of OHLCV DataFrame."""
        # Valid DataFrame should return True
        is_valid, _ = validate_ohlcv_data(sample_ohlcv_dataframe)
        assert is_valid is True

        # Test with empty DataFrame
        is_valid, error_msg = validate_ohlcv_data(pd.DataFrame())
        assert is_valid is False
        assert "empty" in error_msg

        # Test with missing columns
        invalid_df = sample_ohlcv_dataframe.drop('close', axis=1)
        is_valid, error_msg = validate_ohlcv_data(invalid_df)
        assert is_valid is False
        assert "missing required columns" in error_msg

        # Test with non-numeric column
        invalid_df = sample_ohlcv_dataframe.copy()
        invalid_df['close'] = invalid_df['close'].astype(str)
        is_valid, error_msg = validate_ohlcv_data(invalid_df)
        assert is_valid is False
        assert "must contain numeric values" in error_msg

        # Test with invalid type
        is_valid, error_msg = validate_ohlcv_data("invalid")
        assert is_valid is False
        assert "must be either a pandas DataFrame or a list of lists" in error_msg
