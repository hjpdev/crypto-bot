"""
Unit tests for the indicator_service module.
"""
import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame
from app.services.indicator_service import IndicatorService


@pytest.fixture
def sample_ohlcv_data() -> DataFrame:
    """
    Create a sample OHLCV DataFrame for testing.
    """
    # Create a sample DataFrame with 30 days of data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')

    # Create a simple price pattern for testing
    close_prices = [
        100, 102, 104, 103, 105, 107, 108, 109, 110, 112,
        111, 110, 109, 107, 105, 104, 103, 102, 101, 100,
        99, 98, 97, 96, 95, 96, 97, 98, 99, 100
    ]

    # Generate other OHLCV data based on close prices
    data = {
        'open': [close_prices[i-1] if i > 0 else close_prices[0] for i in range(30)],
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'close': close_prices,
        'volume': [1000000 + i * 10000 for i in range(30)]
    }

    df = pd.DataFrame(data, index=dates)
    return df


# Create a larger dataset for MACD testing
@pytest.fixture
def sample_ohlcv_data_large() -> DataFrame:
    """
    Create a larger sample OHLCV DataFrame for testing MACD (which needs more data).
    """
    # Create a sample DataFrame with 60 days of data
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')

    # Create a simple price pattern for testing
    close_prices = []
    for i in range(60):
        # Create a price wave pattern
        price = 100 + 10 * np.sin(i * np.pi / 10)
        close_prices.append(price)

    # Generate other OHLCV data based on close prices
    data = {
        'open': [close_prices[i-1] if i > 0 else close_prices[0] for i in range(60)],
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'close': close_prices,
        'volume': [1000000 + i * 10000 for i in range(60)]
    }

    df = pd.DataFrame(data, index=dates)
    return df


class TestIndicatorService:
    """Test cases for the IndicatorService class."""

    def test_validate_dataframe(self, sample_ohlcv_data):
        """Test validation of input dataframe."""
        # Valid dataframe should return True
        assert IndicatorService.validate_dataframe(sample_ohlcv_data) is True

        # Test with invalid inputs
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            IndicatorService.validate_dataframe([1, 2, 3])

        # Empty dataframe - should be checked first
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            IndicatorService.validate_dataframe(pd.DataFrame())

        # Missing required column
        df_no_close = sample_ohlcv_data.drop('close', axis=1)
        with pytest.raises(ValueError, match="must contain a 'close' column"):
            IndicatorService.validate_dataframe(df_no_close)

    def test_calculate_rsi(self, sample_ohlcv_data):
        """Test RSI calculation."""
        # Calculate RSI with default period
        result = IndicatorService.calculate_rsi(sample_ohlcv_data)

        # Verify result has RSI column
        assert 'rsi' in result.columns

        # Check that RSI values are within the expected range (0-100)
        assert result['rsi'].min() >= 0
        assert result['rsi'].max() <= 100

        # RSI should have NaN values at the beginning (equal to period-1)
        assert result['rsi'].isna().sum() <= 14

        # Test with custom period
        result_short = IndicatorService.calculate_rsi(sample_ohlcv_data, period=5)
        assert result_short['rsi'].isna().sum() <= 5

        # Test with invalid period
        with pytest.raises(ValueError, match="Period must be at least 2"):
            IndicatorService.calculate_rsi(sample_ohlcv_data, period=1)

        # Test with custom column
        sample_ohlcv_data['custom_price'] = sample_ohlcv_data['close'] * 2
        result_custom = IndicatorService.calculate_rsi(
            sample_ohlcv_data, period=14, column='custom_price')
        assert 'rsi' in result_custom.columns

    def test_calculate_macd(self, sample_ohlcv_data_large):
        """Test MACD calculation."""
        # Calculate MACD with default parameters
        result = IndicatorService.calculate_macd(sample_ohlcv_data_large)

        # Verify result has MACD columns
        assert 'MACD_12_26_9' in result.columns
        assert 'MACDs_12_26_9' in result.columns
        assert 'MACDh_12_26_9' in result.columns

        # Test with custom parameters
        result_custom = IndicatorService.calculate_macd(
            sample_ohlcv_data_large, fast=8, slow=21, signal=5)
        assert 'MACD_8_21_5' in result_custom.columns

        # Test with invalid parameters
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            IndicatorService.calculate_macd(sample_ohlcv_data_large, fast=20, slow=10)

        with pytest.raises(ValueError, match="Signal period must be at least 1"):
            IndicatorService.calculate_macd(sample_ohlcv_data_large, signal=0)

        # Test with insufficient data
        small_df = sample_ohlcv_data_large.iloc[:20]  # Only 20 data points
        with pytest.raises(ValueError, match="Not enough data points for MACD calculation"):
            IndicatorService.calculate_macd(small_df, slow=26, signal=9)  # Needs 35 points

    def test_calculate_ema(self, sample_ohlcv_data):
        """Test EMA calculation."""
        # Calculate EMA with period 9
        result = IndicatorService.calculate_ema(sample_ohlcv_data, period=9)

        # Verify result has EMA column
        assert 'ema_9' in result.columns

        # EMA should have fewer NaN values at the beginning (less than period)
        assert result['ema_9'].isna().sum() < 9

        # Test with invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            IndicatorService.calculate_ema(sample_ohlcv_data, period=0)

        # Test with custom column
        result_custom = IndicatorService.calculate_ema(
            sample_ohlcv_data, period=14, column='high')
        assert 'ema_14' in result_custom.columns

    def test_calculate_sma(self, sample_ohlcv_data):
        """Test SMA calculation."""
        # Calculate SMA with period 10
        result = IndicatorService.calculate_sma(sample_ohlcv_data, period=10)

        # Verify result has SMA column
        assert 'sma_10' in result.columns

        # SMA should have NaN values at the beginning (period-1)
        assert result['sma_10'].isna().sum() == 9

        # Test with invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            IndicatorService.calculate_sma(sample_ohlcv_data, period=0)

        # Test with custom column
        result_custom = IndicatorService.calculate_sma(
            sample_ohlcv_data, period=5, column='low')
        assert 'sma_5' in result_custom.columns

    def test_calculate_roc(self, sample_ohlcv_data):
        """Test ROC calculation."""
        # Calculate ROC with period 5
        result = IndicatorService.calculate_roc(sample_ohlcv_data, period=5)

        # Verify result has ROC column
        assert 'roc_5' in result.columns

        # ROC should have NaN values at the beginning (equal to period)
        assert result['roc_5'].isna().sum() == 5

        # Test with invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            IndicatorService.calculate_roc(sample_ohlcv_data, period=0)

        # Test with custom column
        result_custom = IndicatorService.calculate_roc(
            sample_ohlcv_data, period=12, column='high')
        assert 'roc_12' in result_custom.columns

    def test_batch_calculate(self, sample_ohlcv_data_large, sample_ohlcv_data):
        """Test batch calculation of multiple indicators."""
        # Define indicators configuration
        indicators_config = {
            'rsi': {'period': 14, 'column': 'close'},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'ema': [{'period': 9}, {'period': 21}],
            'sma': {'period': 10},
            'roc': {'period': 5}
        }

        # Batch calculate indicators
        result = IndicatorService.batch_calculate(sample_ohlcv_data_large, indicators_config)

        # Verify all expected columns are present
        assert 'rsi' in result.columns
        assert 'MACD_12_26_9' in result.columns
        assert 'MACDs_12_26_9' in result.columns
        assert 'MACDh_12_26_9' in result.columns
        assert 'ema_9' in result.columns
        assert 'ema_21' in result.columns
        assert 'sma_10' in result.columns
        assert 'roc_5' in result.columns

        # Test with empty configuration
        with pytest.raises(ValueError, match="Indicators configuration cannot be empty"):
            IndicatorService.batch_calculate(sample_ohlcv_data_large, {})

        # Test with small dataset - should still work but skip MACD
        small_config = {
            'rsi': {'period': 14, 'column': 'close'},
            'ema': [{'period': 9}, {'period': 21}],
            'sma': {'period': 10},
            'roc': {'period': 5}
        }
        small_result = IndicatorService.batch_calculate(sample_ohlcv_data, small_config)
        assert 'rsi' in small_result.columns
        assert 'ema_9' in small_result.columns
        assert 'ema_21' in small_result.columns
        assert 'sma_10' in small_result.columns
        assert 'roc_5' in small_result.columns

    def test_with_invalid_data(self):
        """Test error handling with invalid data."""
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            IndicatorService.calculate_rsi(None)

        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            IndicatorService.calculate_macd([1, 2, 3])

        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            IndicatorService.batch_calculate("invalid", {})

    def test_different_input_formats(self):
        """Test handling of different input formats."""
        # Create a simple DataFrame with different column name (price instead of close)
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'price': np.linspace(100, 120, 20),
            'volume': np.random.randint(1000, 2000, 20)
        }, index=dates)

        # Calculate RSI using custom column name
        result = IndicatorService.calculate_rsi(df, column='price')
        assert 'rsi' in result.columns

        # Calculate EMA using custom column name
        result = IndicatorService.calculate_ema(df, period=5, column='price')
        assert 'ema_5' in result.columns
