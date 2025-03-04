"""
Unit tests for the indicator_utils module.
"""
import pytest
import pandas as pd
import numpy as np
from app.services.indicator_utils import (
    find_swing_highs,
    find_swing_lows,
    identify_trend,
    smooth_data,
    normalize_indicator
)


@pytest.fixture
def sample_price_data() -> pd.Series:
    """Create a sample price data Series for testing."""
    # Create a sample Series with a wave pattern
    values = []
    for i in range(50):
        # Create a price wave pattern
        price = 100 + 10 * np.sin(i * np.pi / 10)
        values.append(price)

    return pd.Series(values)


@pytest.fixture
def sample_trend_data() -> pd.Series:
    """Create a sample trend data Series for testing."""
    # Create a sample Series with an up-trend, down-trend and sideways pattern
    values = [
        # Uptrend
        100, 101, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107,
        # Downtrend
        106, 105, 106, 104, 105, 103, 104, 102, 103, 101, 102, 100,
        # Sideways
        99, 100, 99, 101, 100, 102, 101, 100, 99, 100, 101, 100
    ]

    return pd.Series(values)


class TestIndicatorUtils:
    """Test cases for the indicator utility functions."""

    def test_find_swing_highs(self, sample_price_data):
        """Test finding swing high points."""
        # Find swing highs with default window
        swing_highs = find_swing_highs(sample_price_data)

        # Should find some swing highs
        assert len(swing_highs) > 0

        # Each swing high should be a local maximum
        for i in swing_highs:
            if i > 0 and i < len(sample_price_data) - 1:
                # Simple check: value should be higher than immediate neighbors
                assert sample_price_data[i] > sample_price_data[i-1]
                assert sample_price_data[i] > sample_price_data[i+1]

        # Test with custom window
        swing_highs_larger = find_swing_highs(sample_price_data, window=10)

        # Larger window should find fewer swing highs
        assert len(swing_highs_larger) <= len(swing_highs)

        # Test invalid window
        with pytest.raises(ValueError, match="Window must be at least 2"):
            find_swing_highs(sample_price_data, window=1)

        # Test with tiny dataset
        tiny_data = pd.Series([1, 2, 3])
        assert find_swing_highs(tiny_data, window=2) == []

    def test_find_swing_lows(self, sample_price_data):
        """Test finding swing low points."""
        # Find swing lows with default window
        swing_lows = find_swing_lows(sample_price_data)

        # Should find some swing lows
        assert len(swing_lows) > 0

        # Each swing low should be a local minimum
        for i in swing_lows:
            if i > 0 and i < len(sample_price_data) - 1:
                # Simple check: value should be lower than immediate neighbors
                assert sample_price_data[i] < sample_price_data[i-1]
                assert sample_price_data[i] < sample_price_data[i+1]

        # Test with custom window
        swing_lows_larger = find_swing_lows(sample_price_data, window=10)

        # Larger window should find fewer swing lows
        assert len(swing_lows_larger) <= len(swing_lows)

        # Test invalid window
        with pytest.raises(ValueError, match="Window must be at least 2"):
            find_swing_lows(sample_price_data, window=1)

        # Test with tiny dataset
        tiny_data = pd.Series([3, 2, 1])
        assert find_swing_lows(tiny_data, window=2) == []

    def test_identify_trend(self, sample_trend_data):
        """Test trend identification."""
        # Identify trend with default window
        trend = identify_trend(sample_trend_data)

        # Should have some non-zero values indicating trends
        assert (trend != 0).any()

        # Early part of data should have uptrends
        uptrend_count = (trend.iloc[:13] == 1).sum()
        assert uptrend_count > 0

        # Middle part should have downtrends
        downtrend_count = (trend.iloc[13:25] == -1).sum()
        assert downtrend_count > 0

        # Test with custom window
        trend_small = identify_trend(sample_trend_data, window=5)
        assert len(trend_small) == len(sample_trend_data)

        # Test with numpy array input
        trend_np = identify_trend(np.array(sample_trend_data))
        assert isinstance(trend_np, pd.Series)
        assert len(trend_np) == len(sample_trend_data)

        # Test with invalid window
        with pytest.raises(ValueError, match="Data length must be at least 14"):
            identify_trend(pd.Series([1, 2, 3, 4, 5]))

    def test_smooth_data(self, sample_price_data):
        """Test data smoothing methods."""
        # Test different smoothing methods
        smoothing_methods = ['sma', 'ema', 'wma', 'hull']

        for method in smoothing_methods:
            smoothed = smooth_data(sample_price_data, method=method, period=5)

            # Smoothed data should have the same length
            assert len(smoothed) == len(sample_price_data)

            # Early values should be NaN (except for LOWESS)
            assert smoothed.iloc[:5].isna().any()

            # Later values should be populated
            assert not smoothed.iloc[5:].isna().all()

        # Test LOWESS smoothing
        lowess_smoothed = smooth_data(sample_price_data, method='lowess', period=5)
        assert len(lowess_smoothed) == len(sample_price_data)

        # Test with numpy array input
        np_smoothed = smooth_data(np.array(sample_price_data), method='sma', period=5)
        assert isinstance(np_smoothed, pd.Series)

        # Test with invalid parameters
        with pytest.raises(ValueError, match="Period must be at least 2"):
            smooth_data(sample_price_data, period=1)

        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smooth_data(sample_price_data, method='invalid_method')

        with pytest.raises(ValueError, match="Data length must be at least 5"):
            smooth_data(pd.Series([1, 2, 3]), period=5)

    def test_normalize_indicator(self, sample_price_data):
        """Test indicator normalization methods."""
        # Test different normalization methods
        norm_methods = ['minmax', 'zscore', 'tanh', 'sigmoid']

        for method in norm_methods:
            normalized = normalize_indicator(sample_price_data, method=method)

            # Normalized data should have the same length
            assert len(normalized) == len(sample_price_data)

            # Normalized data should not have NaN values
            assert not normalized.isna().any()

            # Check specific ranges based on method
            if method == 'minmax':
                assert normalized.min() >= 0
                assert normalized.max() <= 1
            elif method == 'tanh':
                assert normalized.min() >= -1
                assert normalized.max() <= 1
            elif method == 'sigmoid':
                assert normalized.min() >= 0
                assert normalized.max() <= 1

        # Test with numpy array input
        np_normalized = normalize_indicator(np.array(sample_price_data), method='minmax')
        assert isinstance(np_normalized, pd.Series)

        # Test edge cases
        constant_data = pd.Series([5, 5, 5, 5, 5])

        # MinMax normalization of constant data should return 0.5
        minmax_constant = normalize_indicator(constant_data, method='minmax')
        assert (minmax_constant == 0.5).all()

        # Z-score normalization of constant data should return 0
        zscore_constant = normalize_indicator(constant_data, method='zscore')
        assert (zscore_constant == 0).all()

        # Test with invalid method
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_indicator(sample_price_data, method='invalid_method')

        # Test with empty data
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            normalize_indicator(pd.Series([]))

    def test_with_known_patterns(self):
        """Test utilities with known patterns to verify results."""
        # Test swing detection with simple pattern
        simple_pattern = pd.Series([10, 20, 15, 30, 25, 40, 35, 50, 45, 60])

        # Even indices (0, 2, 4, 6, 8) should be swing lows, odd indices should be swing highs
        # But our algorithm requires a window around points, so not all will be detected
        swing_highs = find_swing_highs(simple_pattern, window=2)
        swing_lows = find_swing_lows(simple_pattern, window=2)

        # Check that at least some expected points are detected
        assert 1 in swing_highs  # 20 should be a swing high
        assert 3 in swing_highs  # 30 should be a swing high
        assert 2 in swing_lows   # 15 should be a swing low
        assert 4 in swing_lows   # 25 should be a swing low

        # Test trend identification with simple uptrend
        uptrend = pd.Series([
            10, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17, 19, 18, 20
        ])
        trend = identify_trend(uptrend, window=6)

        # Should identify uptrend in later elements
        assert (trend.iloc[10:] == 1).any()

    def test_utility_combinations(self, sample_price_data):
        """Test using utilities together in common patterns."""
        # Smooth the data first
        smoothed = smooth_data(sample_price_data, method='ema', period=5)

        # Find swing points on smoothed data
        swing_highs = find_swing_highs(smoothed, window=3)
        swing_lows = find_swing_lows(smoothed, window=3)

        # Check that smoothing doesn't eliminate all swing points
        assert len(swing_highs) > 0
        assert len(swing_lows) > 0

        # Normalize the data
        normalized = normalize_indicator(smoothed, method='minmax')

        # Check that normalization preserves the shape
        assert normalized.argmax() == smoothed.argmax()
        assert normalized.argmin() == smoothed.argmin()
