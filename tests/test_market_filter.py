"""
Tests for the market filter module.

This module tests the various filtering functions provided by the MarketFilter class,
including tests for individual filters, combined filtering, and edge cases.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from app.services.market_filter import MarketFilter
from app.services.exchange_service import ExchangeService


@pytest.fixture
def mock_exchange_service():
    """Create a mock exchange service for testing."""
    mock_service = MagicMock(spec=ExchangeService)
    return mock_service


@pytest.fixture
def market_filter(mock_exchange_service):
    """Create a MarketFilter instance with a mock exchange service."""
    return MarketFilter(mock_exchange_service)


@pytest.fixture
def mock_ticker_data():
    """Create mock ticker data for testing."""
    return {
        'BTC/USD': {
            'symbol': 'BTC/USD',
            'last': 50000.0,
            'quoteVolume': 1000000.0,
            'marketCap': 900000000000.0,
            'spread': 0.1,
        },
        'ETH/USD': {
            'symbol': 'ETH/USD',
            'last': 3000.0,
            'quoteVolume': 500000.0,
            'marketCap': 350000000000.0,
            'spread': 0.15,
        },
        'XRP/USD': {
            'symbol': 'XRP/USD',
            'last': 0.5,
            'quoteVolume': 100000.0,
            'marketCap': 25000000000.0,
            'spread': 0.3,
        },
        'DOGE/USD': {
            'symbol': 'DOGE/USD',
            'last': 0.1,
            'quoteVolume': 50000.0,
            'marketCap': 15000000000.0,
            'spread': 0.4,
        },
        'SHIB/USD': {
            'symbol': 'SHIB/USD',
            'last': 0.00001,
            'quoteVolume': 10000.0,
            'marketCap': 5000000000.0,
            'spread': 0.5,
        },
        'BTC/USDT': {
            'symbol': 'BTC/USDT',
            'last': 50100.0,
            'quoteVolume': 1100000.0,
            'marketCap': 900000000000.0,
            'spread': 0.12,
        },
        'ETH/EUR': {
            'symbol': 'ETH/EUR',
            'last': 2800.0,
            'quoteVolume': 400000.0,
            'marketCap': 350000000000.0,
            'spread': 0.18,
        },
    }


@pytest.fixture
def mock_volatility_data():
    """Create mock volatility data for testing."""
    return {
        'BTC/USD': 3.5,
        'ETH/USD': 5.2,
        'XRP/USD': 8.7,
        'DOGE/USD': 12.4,
        'SHIB/USD': 18.9,
        'BTC/USDT': 3.6,
        'ETH/EUR': 5.3,
    }


class TestMarketFilter:
    """Test suite for the MarketFilter class."""

    def test_filter_by_market_cap(self, market_filter, mock_exchange_service, mock_ticker_data):
        """Test filtering by market cap."""
        # Setup
        symbols = list(mock_ticker_data.keys())
        mock_exchange_service.get_ticker_batch.return_value = mock_ticker_data

        # Set up the cache to avoid actual API calls
        market_filter._market_data_cache = mock_ticker_data
        market_filter._cache_timestamp = datetime.now()

        # Test with high market cap threshold (only BTC and ETH should pass)
        filtered = market_filter.filter_by_market_cap(symbols, 100000000000.0)
        assert len(filtered) == 4  # BTC/USD, ETH/USD, BTC/USDT, ETH/EUR
        assert 'BTC/USD' in filtered
        assert 'ETH/USD' in filtered
        assert 'BTC/USDT' in filtered
        assert 'ETH/EUR' in filtered
        assert 'XRP/USD' not in filtered

        # Test with lower market cap threshold (most should pass)
        filtered = market_filter.filter_by_market_cap(symbols, 10000000000.0)
        assert len(filtered) == 6  # All except SHIB/USD
        assert 'SHIB/USD' not in filtered

        # Test with very low market cap threshold (all should pass)
        filtered = market_filter.filter_by_market_cap(symbols, 1000000000.0)
        assert len(filtered) == len(symbols)

    def test_filter_by_volume(self, market_filter, mock_exchange_service, mock_ticker_data):
        """Test filtering by trading volume."""
        # Setup
        symbols = list(mock_ticker_data.keys())
        mock_exchange_service.get_ticker_batch.return_value = mock_ticker_data

        # Set up the cache to avoid actual API calls
        market_filter._market_data_cache = mock_ticker_data
        market_filter._cache_timestamp = datetime.now()

        # Test with high volume threshold
        filtered = market_filter.filter_by_volume(symbols, 800000.0)
        assert len(filtered) == 2  # BTC/USD, BTC/USDT
        assert 'BTC/USD' in filtered
        assert 'BTC/USDT' in filtered

        # Test with medium volume threshold
        filtered = market_filter.filter_by_volume(symbols, 100000.0)
        assert len(filtered) == 5  # BTC/USD, ETH/USD, XRP/USD, BTC/USDT, ETH/EUR
        assert 'BTC/USD' in filtered
        assert 'ETH/USD' in filtered
        assert 'XRP/USD' in filtered
        assert 'BTC/USDT' in filtered
        assert 'ETH/EUR' in filtered

        # Test with low volume threshold (most should pass)
        filtered = market_filter.filter_by_volume(symbols, 10000.0)
        assert len(filtered) == len(symbols)

    def test_filter_by_spread(self, market_filter, mock_exchange_service, mock_ticker_data):
        """Test filtering by spread percentage."""
        # Setup
        symbols = list(mock_ticker_data.keys())
        mock_exchange_service.get_ticker_batch.return_value = mock_ticker_data

        # Set up the cache to avoid actual API calls
        market_filter._market_data_cache = mock_ticker_data
        market_filter._cache_timestamp = datetime.now()

        # Test with low spread threshold (only the lowest spread coins should pass)
        filtered = market_filter.filter_by_spread(symbols, 0.12)
        assert len(filtered) == 2  # BTC/USD, BTC/USDT
        assert 'BTC/USD' in filtered
        assert 'BTC/USDT' in filtered

        # Test with medium spread threshold
        filtered = market_filter.filter_by_spread(symbols, 0.2)
        assert len(filtered) == 4  # BTC/USD, ETH/USD, BTC/USDT, ETH/EUR
        assert 'BTC/USD' in filtered
        assert 'ETH/USD' in filtered
        assert 'BTC/USDT' in filtered
        assert 'ETH/EUR' in filtered

        # Test with high spread threshold (most should pass)
        filtered = market_filter.filter_by_spread(symbols, 0.5)
        assert len(filtered) == len(symbols)

    def test_filter_by_volatility(self, market_filter, mock_exchange_service, mock_volatility_data):
        """Test filtering by volatility range."""
        # Setup
        symbols = list(mock_volatility_data.keys())

        # Mock the _get_volatility_data method
        with patch.object(market_filter, '_get_volatility_data', return_value=mock_volatility_data):
            # Test with narrow volatility range (only BTC should pass)
            filtered = market_filter.filter_by_volatility(symbols, 3.0, 4.0)
            assert len(filtered) == 2  # BTC/USD, BTC/USDT
            assert 'BTC/USD' in filtered
            assert 'BTC/USDT' in filtered

            # Test with medium volatility range (BTC and ETH should pass)
            filtered = market_filter.filter_by_volatility(symbols, 3.0, 6.0)
            assert len(filtered) == 4  # BTC/USD, ETH/USD, BTC/USDT, ETH/EUR

            # Test with high volatility range (most should pass)
            filtered = market_filter.filter_by_volatility(symbols, 3.0, 15.0)
            assert len(filtered) == 6  # All except SHIB/USD
            assert 'SHIB/USD' not in filtered

            # Test with very wide range (all should pass)
            filtered = market_filter.filter_by_volatility(symbols, 0.0, 20.0)
            assert len(filtered) == len(symbols)

    def test_filter_by_allowed_quote(self, market_filter):
        """Test filtering by allowed quote currencies."""
        # Setup
        symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'BTC/USDT', 'ETH/EUR', 'SOL/JPY', 'LTC/GBP']

        # Test with USD only
        filtered = market_filter.filter_by_allowed_quote(symbols, ['USD'])
        assert len(filtered) == 3
        assert 'BTC/USD' in filtered
        assert 'ETH/USD' in filtered
        assert 'XRP/USD' in filtered

        # Test with USD and USDT
        filtered = market_filter.filter_by_allowed_quote(symbols, ['USD', 'USDT'])
        assert len(filtered) == 4
        assert 'BTC/USDT' in filtered

        # Test with EUR
        filtered = market_filter.filter_by_allowed_quote(symbols, ['EUR'])
        assert len(filtered) == 1
        assert 'ETH/EUR' in filtered

        # Test with quote currency not in the list
        filtered = market_filter.filter_by_allowed_quote(symbols, ['JPY'])
        assert len(filtered) == 1
        assert 'SOL/JPY' in filtered

        # Test with multiple currencies
        filtered = market_filter.filter_by_allowed_quote(symbols, ['USD', 'EUR', 'GBP'])
        assert len(filtered) == 5

    def test_apply_all_filters(
        self,
        market_filter,
        mock_exchange_service,
        mock_ticker_data,
        mock_volatility_data,
    ):
        """Test applying all filters combined."""
        # Setup
        symbols = list(mock_ticker_data.keys())
        mock_exchange_service.get_ticker_batch.return_value = mock_ticker_data

        # Set up the cache to avoid actual API calls
        market_filter._market_data_cache = mock_ticker_data
        market_filter._cache_timestamp = datetime.now()

        # Mock the _get_volatility_data method
        with patch.object(market_filter, '_get_volatility_data', return_value=mock_volatility_data):
            # Test with multiple filters
            config = {
                'allowed_quotes': ['USD', 'USDT'],
                'min_market_cap': 500000000000.0,  # Increased to filter out ETH/USD
                'min_volume': 500000.0,
                'max_spread': 0.2,
                'min_volatility': 3.0,
                'max_volatility': 6.0,
            }

            filtered = market_filter.apply_all_filters(symbols, config)

            # Only BTC/USD and BTC/USDT should pass all filters
            assert len(filtered) == 2
            assert 'BTC/USD' in filtered
            assert 'BTC/USDT' in filtered
            assert 'ETH/USD' not in filtered  # ETH should be filtered out by market cap now

            # Test with less restrictive filters
            config = {
                'allowed_quotes': ['USD', 'USDT'],
                'min_market_cap': 20000000000.0,
                'min_volume': 50000.0,
                'max_spread': 0.4,
                'min_volatility': 0.0,
                'max_volatility': 15.0,
            }

            filtered = market_filter.apply_all_filters(symbols, config)
            assert len(filtered) == 4  # BTC/USD, ETH/USD, XRP/USD, BTC/USDT

            # Test with only one filter
            config = {
                'allowed_quotes': ['USD', 'USDT'],
            }

            filtered = market_filter.apply_all_filters(symbols, config)
            assert len(filtered) == 6  # All USD and USDT pairs

    def test_empty_input(self, market_filter):
        """Test with empty input list."""
        # All filter methods should return empty list for empty input
        assert market_filter.filter_by_market_cap([], 1000000.0) == []
        assert market_filter.filter_by_volume([], 1000.0) == []
        assert market_filter.filter_by_spread([], 0.1) == []
        assert market_filter.filter_by_volatility([], 1.0, 5.0) == []
        assert market_filter.filter_by_allowed_quote([], ['USD']) == []
        assert market_filter.apply_all_filters([], {'min_volume': 1000.0}) == []

    def test_all_filtered_out(self, market_filter, mock_exchange_service, mock_ticker_data):
        """Test when all symbols are filtered out."""
        # Setup
        symbols = list(mock_ticker_data.keys())
        mock_exchange_service.get_ticker_batch.return_value = mock_ticker_data

        # Set up the cache to avoid actual API calls
        market_filter._market_data_cache = mock_ticker_data
        market_filter._cache_timestamp = datetime.now()

        # Test with unreachable threshold
        filtered = market_filter.filter_by_market_cap(
            symbols, 1000000000000000.0
        )  # Unrealistic market cap
        assert filtered == []

        filtered = market_filter.filter_by_volume(symbols, 10000000000.0)  # Unrealistic volume
        assert filtered == []

    def test_cache_behavior(self, market_filter, mock_exchange_service, mock_ticker_data):
        """Test that caching works correctly."""
        # Setup
        symbols = list(mock_ticker_data.keys())
        mock_exchange_service.get_ticker_batch.return_value = mock_ticker_data

        # First call should fetch from exchange
        market_filter.filter_by_market_cap(symbols, 100000000000.0)
        assert mock_exchange_service.get_ticker_batch.called
        mock_exchange_service.get_ticker_batch.reset_mock()

        # Second call should use cache
        market_filter.filter_by_market_cap(symbols, 50000000000.0)
        assert not mock_exchange_service.get_ticker_batch.called

        # Expire the cache
        market_filter._cache_timestamp = datetime.now() - timedelta(seconds=301)

        # Call should fetch from exchange again
        market_filter.filter_by_market_cap(symbols, 100000000000.0)
        assert mock_exchange_service.get_ticker_batch.called

    def test_market_data_error_handling(self, market_filter, mock_exchange_service):
        """Test error handling when getting market data."""
        # Setup
        symbols = ['BTC/USD', 'ETH/USD']
        mock_exchange_service.get_ticker_batch.side_effect = Exception("API error")
        mock_exchange_service.get_ticker.side_effect = [
            {
                'symbol': 'BTC/USD',
                'last': 50000.0,
                'quoteVolume': 1000000.0,
                'marketCap': 900000000000.0,
            },
            Exception("Individual API error"),
        ]

        # Should handle batch error and fall back to individual fetches
        filtered = market_filter.filter_by_market_cap(symbols, 100000000000.0)

        # Should still get results for BTC/USD
        assert len(filtered) == 1
        assert 'BTC/USD' in filtered

    def test_volatility_calculation_error_handling(self, market_filter, mock_exchange_service):
        """Test error handling during volatility calculation."""
        # Setup
        symbols = ['BTC/USD', 'ETH/USD']

        # Mock fetch_ohlcv to succeed for BTC but fail for ETH
        def mock_fetch_ohlcv(symbol, **kwargs):
            if symbol == 'BTC/USD':
                # Return mock OHLCV data for BTC
                return [
                    [1614556800000, 45000, 46000, 44000, 45500, 1000],
                    [1614643200000, 45500, 47000, 45000, 46800, 1200],
                    [1614729600000, 46800, 48000, 46500, 47500, 1100],
                ]
            else:
                # Simulate error for ETH
                raise Exception("OHLCV data fetch error")

        mock_exchange_service.fetch_ohlcv.side_effect = mock_fetch_ohlcv

        # Should handle the error for ETH but still calculate for BTC
        with patch.object(
            market_filter, '_get_volatility_data', wraps=market_filter._get_volatility_data
        ):
            volatility_data = market_filter._get_volatility_data(symbols)

            assert 'BTC/USD' in volatility_data
            assert volatility_data['ETH/USD'] is None

            # Use the volatility data for filtering
            filtered = market_filter.filter_by_volatility(symbols, 0, 10, '1d', 2)

            # Only BTC should pass since ETH has no data
            assert len(filtered) == 1
            assert 'BTC/USD' in filtered
