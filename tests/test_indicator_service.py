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

        # RSI should have some NaN values at the beginning due to the calculation window
        assert result['rsi'].iloc[:14].isna().any()

        # Later values should be populated
        assert not result['rsi'].iloc[14:].isna().all()

        # Test with custom period
        result_short = IndicatorService.calculate_rsi(sample_ohlcv_data, period=5)
        assert not result_short['rsi'].iloc[5:].isna().all()

        # Invalid period should raise ValueError
        with pytest.raises(ValueError, match="Period must be at least 2"):
            IndicatorService.calculate_rsi(sample_ohlcv_data, period=1)

    def test_calculate_macd(self, sample_ohlcv_data_large):
        """Test MACD calculation."""
        # Calculate MACD with default parameters
        result = IndicatorService.calculate_macd(sample_ohlcv_data_large)

        # Verify result has MACD columns
        assert 'MACD_12_26_9' in result.columns
        assert 'MACDs_12_26_9' in result.columns
        assert 'MACDh_12_26_9' in result.columns

        # MACD should have some NaN values at the beginning due to the calculation window
        assert result['MACD_12_26_9'].iloc[:26].isna().any()

        # Later values should be populated
        assert not result['MACD_12_26_9'].iloc[26:].isna().all()

        # Test with custom parameters
        result_custom = IndicatorService.calculate_macd(
            sample_ohlcv_data_large, fast=8, slow=21, signal=5
        )
        assert 'MACD_8_21_5' in result_custom.columns

        # Invalid parameters should raise ValueError
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            IndicatorService.calculate_macd(sample_ohlcv_data_large, fast=20, slow=20)

    def test_calculate_ema(self, sample_ohlcv_data):
        """Test EMA calculation."""
        # Calculate EMA with period=10
        result = IndicatorService.calculate_ema(sample_ohlcv_data, period=10)

        # Verify result has EMA column
        assert 'ema_10' in result.columns

        # EMA should have some NaN values at the beginning due to the calculation window
        assert result['ema_10'].iloc[:10].isna().any()

        # Later values should be populated
        assert not result['ema_10'].iloc[10:].isna().all()

        # Invalid period should raise ValueError
        with pytest.raises(ValueError, match="Period must be at least 1"):
            IndicatorService.calculate_ema(sample_ohlcv_data, period=0)

    def test_calculate_sma(self, sample_ohlcv_data):
        """Test SMA calculation."""
        # Calculate SMA with period=10
        result = IndicatorService.calculate_sma(sample_ohlcv_data, period=10)

        # Verify result has SMA column
        assert 'sma_10' in result.columns

        # SMA should have some NaN values at the beginning due to the calculation window
        assert result['sma_10'].iloc[:10].isna().any()

        # Later values should be populated
        assert not result['sma_10'].iloc[10:].isna().all()

        # Invalid period should raise ValueError
        with pytest.raises(ValueError, match="Period must be at least 1"):
            IndicatorService.calculate_sma(sample_ohlcv_data, period=0)

    def test_calculate_roc(self, sample_ohlcv_data):
        """Test ROC calculation."""
        # Calculate ROC with period=10
        result = IndicatorService.calculate_roc(sample_ohlcv_data, period=10)

        # Verify result has ROC column
        assert 'roc_10' in result.columns

        # ROC should have some NaN values at the beginning due to the calculation window
        assert result['roc_10'].iloc[:10].isna().any()

        # Later values should be populated
        assert not result['roc_10'].iloc[10:].isna().all()

        # Invalid period should raise ValueError
        with pytest.raises(ValueError, match="Period must be at least 1"):
            IndicatorService.calculate_roc(sample_ohlcv_data, period=0)

    def test_calculate_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        # Calculate Bollinger Bands with default parameters
        result = IndicatorService.calculate_bollinger_bands(sample_ohlcv_data)

        # Verify result has Bollinger Bands columns
        assert 'BBL_20_2.0' in result.columns  # Lower band
        assert 'BBM_20_2.0' in result.columns  # Middle band (SMA)
        assert 'BBU_20_2.0' in result.columns  # Upper band
        assert 'BBB_20_2.0' in result.columns  # Band width

        # Bands should have some NaN values at the beginning due to the calculation window
        assert result['BBL_20_2.0'].iloc[:20].isna().any()

        # Later values should be populated
        assert not result['BBL_20_2.0'].iloc[20:].isna().all()

        # Upper band should be greater than middle band
        valid_indices = result['BBU_20_2.0'].notna()
        assert (
            result.loc[valid_indices, 'BBU_20_2.0'] >= result.loc[valid_indices, 'BBM_20_2.0']
        ).all()

        # Lower band should be less than middle band
        assert (
            result.loc[valid_indices, 'BBL_20_2.0'] <= result.loc[valid_indices, 'BBM_20_2.0']
        ).all()

        # Test with custom parameters
        result_custom = IndicatorService.calculate_bollinger_bands(
            sample_ohlcv_data, period=10, std_dev=3
        )
        assert 'BBL_10_3.0' in result_custom.columns

        # Invalid parameters should raise ValueError
        with pytest.raises(ValueError, match="Period must be at least 2"):
            IndicatorService.calculate_bollinger_bands(sample_ohlcv_data, period=1)

        with pytest.raises(ValueError, match="Standard deviation must be non-negative"):
            IndicatorService.calculate_bollinger_bands(sample_ohlcv_data, std_dev=-1)

    def test_calculate_atr(self, sample_ohlcv_data):
        """Test ATR calculation."""
        # Calculate ATR with default period
        result = IndicatorService.calculate_atr(sample_ohlcv_data)

        # Verify result has ATR column
        assert 'atr' in result.columns

        # ATR should have some NaN values at the beginning due to the calculation window
        assert result['atr'].iloc[:14].isna().any()

        # Later values should be populated
        assert not result['atr'].iloc[14:].isna().all()

        # ATR should be non-negative
        valid_indices = result['atr'].notna()
        assert (result.loc[valid_indices, 'atr'] >= 0).all()

        # Test with custom period
        result_custom = IndicatorService.calculate_atr(sample_ohlcv_data, period=7)
        assert not result_custom['atr'].iloc[7:].isna().all()

        # Invalid period should raise ValueError
        with pytest.raises(ValueError, match="Period must be at least 1"):
            IndicatorService.calculate_atr(sample_ohlcv_data, period=0)

    def test_calculate_adx(self, sample_ohlcv_data):
        """Test ADX calculation."""
        # Calculate ADX with default period
        result = IndicatorService.calculate_adx(sample_ohlcv_data)

        # Verify result has ADX columns
        assert 'ADX_14' in result.columns
        assert 'DMP_14' in result.columns  # Positive Directional Movement
        assert 'DMN_14' in result.columns  # Negative Directional Movement

        # ADX should have some NaN values at the beginning due to the calculation window
        assert result['ADX_14'].iloc[:28].isna().any()  # ADX needs 2*period data points

        # Later values should be populated
        assert not result['ADX_14'].iloc[28:].isna().all()

        # ADX should be between 0 and 100
        valid_indices = result['ADX_14'].notna()
        assert (result.loc[valid_indices, 'ADX_14'] >= 0).all()
        assert (result.loc[valid_indices, 'ADX_14'] <= 100).all()

        # Test with custom period
        result_custom = IndicatorService.calculate_adx(sample_ohlcv_data, period=7)
        assert 'ADX_7' in result_custom.columns

        # Invalid period should raise ValueError
        with pytest.raises(ValueError, match="Period must be at least 2"):
            IndicatorService.calculate_adx(sample_ohlcv_data, period=1)

    def test_calculate_obv(self, sample_ohlcv_data):
        """Test OBV calculation."""
        # Calculate OBV
        result = IndicatorService.calculate_obv(sample_ohlcv_data)

        # Verify result has OBV column
        assert 'obv' in result.columns

        # OBV should not have NaN values
        assert not result['obv'].isna().any()

        # Test with missing volume column
        df_no_volume = sample_ohlcv_data.drop('volume', axis=1)
        with pytest.raises(ValueError, match="must contain a 'volume' column"):
            IndicatorService.calculate_obv(df_no_volume)

    def test_calculate_vwap(self, sample_ohlcv_data):
        """Test VWAP calculation."""
        # Calculate VWAP
        result = IndicatorService.calculate_vwap(sample_ohlcv_data)

        # Verify result has VWAP column
        assert 'vwap' in result.columns

        # VWAP should not have NaN values
        assert not result['vwap'].isna().any()

        # VWAP should be within the range of high and low prices
        assert (result['vwap'] >= result['low'].min()).all()
        assert (result['vwap'] <= result['high'].max()).all()

        # Test with missing required columns
        df_no_high = sample_ohlcv_data.drop('high', axis=1)
        with pytest.raises(ValueError, match="must contain a 'high' column"):
            IndicatorService.calculate_vwap(df_no_high)

    def test_calculate_support_resistance(self, sample_ohlcv_data):
        """Test support and resistance calculation."""
        # Calculate support and resistance levels
        result = IndicatorService.calculate_support_resistance(sample_ohlcv_data)

        # Verify result has support and resistance columns
        assert 'support' in result.columns
        assert 'resistance' in result.columns

        # Most values should be zero (only swing points are marked)
        assert (result['support'] == 0).sum() > 0
        assert (result['resistance'] == 0).sum() > 0

        # Support values should be non-zero at some points
        assert (result['support'] > 0).sum() > 0

        # Resistance values should be non-zero at some points
        assert (result['resistance'] > 0).sum() > 0

        # Test with custom lookback period
        result_custom = IndicatorService.calculate_support_resistance(
            sample_ohlcv_data, lookback=7
        )
        assert 'support' in result_custom.columns

        # Invalid lookback should raise ValueError
        with pytest.raises(ValueError, match="Lookback period must be at least 3"):
            IndicatorService.calculate_support_resistance(sample_ohlcv_data, lookback=2)

    def test_calculate_multi_timeframe(self, sample_ohlcv_data_large):
        """Test multi-timeframe indicator calculation."""
        # Create dataframes for different timeframes
        # For testing purposes, we'll use resampled versions of the same data
        dataframes_dict = {
            '1h': sample_ohlcv_data_large,
            '4h': sample_ohlcv_data_large.iloc[::4].copy(),  # Every 4th row
            '1d': sample_ohlcv_data_large.iloc[::24].copy()  # Every 24th row
        }

        # Calculate RSI across multiple timeframes
        result = IndicatorService.calculate_multi_timeframe(
            dataframes_dict,
            IndicatorService.calculate_rsi,
            period=14
        )

        # Verify result contains all timeframes
        assert set(result.keys()) == {'1h', '4h', '1d'}

        # Each result should have an RSI column
        assert 'rsi' in result['1h'].columns
        assert 'rsi' in result['4h'].columns
        assert 'rsi' in result['1d'].columns

        # Test with empty dictionary
        with pytest.raises(ValueError, match="Dataframes dictionary cannot be empty"):
            IndicatorService.calculate_multi_timeframe({}, IndicatorService.calculate_rsi)

    def test_detect_divergence(self, sample_ohlcv_data):
        """Test divergence detection."""
        # Calculate RSI for the sample data
        rsi_df = IndicatorService.calculate_rsi(sample_ohlcv_data)

        # Create artificially divergent data for testing
        price_data = sample_ohlcv_data['close']
        rsi_data = rsi_df['rsi']

        # Detect divergence
        result = IndicatorService.detect_divergence(price_data, rsi_data)

        # Verify result columns
        assert 'price' in result.columns
        assert 'indicator' in result.columns
        assert 'bullish_divergence' in result.columns
        assert 'bearish_divergence' in result.columns

        # Test with invalid inputs
        with pytest.raises(ValueError, match="Price and indicator data must be pandas Series"):
            IndicatorService.detect_divergence([1, 2, 3], rsi_data)

        # Test with mismatched lengths
        with pytest.raises(ValueError, match="Price and indicator data must have the same length"):
            IndicatorService.detect_divergence(price_data.iloc[:-5], rsi_data)

        # Test with invalid window
        with pytest.raises(ValueError, match="Window must be at least 3"):
            IndicatorService.detect_divergence(price_data, rsi_data, window=2)

    def test_batch_calculate(self, sample_ohlcv_data_large, sample_ohlcv_data):
        """Test batch calculation of multiple indicators."""
        # Define indicators to calculate
        indicators_config = {
            'rsi': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'ema': [{'period': 9}, {'period': 21}],
            'sma': {'period': 50},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14},
            'adx': {'period': 14},
            'obv': {},
            'vwap': {}
        }

        # Calculate all indicators in one batch for the large dataset
        result = IndicatorService.batch_calculate(sample_ohlcv_data_large, indicators_config)

        # Verify that all expected columns are present
        assert 'rsi' in result.columns
        assert 'MACD_12_26_9' in result.columns
        assert 'ema_9' in result.columns
        assert 'ema_21' in result.columns
        assert 'sma_50' in result.columns
        assert 'BBU_20_2.0' in result.columns
        assert 'atr' in result.columns
        assert 'ADX_14' in result.columns
        assert 'obv' in result.columns
        assert 'vwap' in result.columns

        # Test with empty config
        with pytest.raises(ValueError, match="Indicators configuration cannot be empty"):
            IndicatorService.batch_calculate(sample_ohlcv_data, {})

    def test_with_invalid_data(self):
        """Test error handling with invalid input data."""
        # Create an empty dataframe
        empty_df = pd.DataFrame()

        # Methods should raise ValueErrors for empty dataframes
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            IndicatorService.calculate_rsi(empty_df)

        # Create a dataframe with missing required columns
        invalid_df = pd.DataFrame({'price': [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain a 'close' column"):
            IndicatorService.calculate_bollinger_bands(invalid_df)

    def test_different_input_formats(self):
        """Test handling of different input formats."""
        # Create a simple DataFrame with only the required columns
        minimal_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 3,
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115] * 3,
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105] * 3,
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000] * 3
        })

        # Test a basic indicator calculation
        result = IndicatorService.calculate_rsi(minimal_df, period=5)
        assert 'rsi' in result.columns

        # Test a more complex indicator
        result = IndicatorService.calculate_bollinger_bands(minimal_df, period=5)
        assert 'BBU_5_2.0' in result.columns
