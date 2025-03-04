"""
Tests for the exchange service.

This module tests the exchange service's functionality, including initialization,
API calls, rate limiting, error handling, and caching.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ccxt.base.errors import RateLimitExceeded
import functools

from app.services.exchange_service import ExchangeService
from app.services.exchange_rate_limiter import RateLimiter


class TestExchangeService:
    """Tests for the ExchangeService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_exchange = Mock()
        self.mock_exchange.id = 'binance'
        self.mock_exchange.name = 'Binance'
        self.mock_exchange.has = {
            'fetchOHLCV': True,
            'fetchOrderBook': True,
        }
        self.mock_exchange.timeframes = {'1m': '1m', '1h': '1h', '1d': '1d'}

        # Create a patcher for ccxt.exchanges
        self.exchanges_patcher = patch('app.services.exchange_service.ccxt.exchanges',
                                       ['binance', 'coinbase', 'kraken'])
        self.mock_exchanges = self.exchanges_patcher.start()

        # Create a patcher for getattr
        self.patcher = patch('app.services.exchange_service.getattr')
        self.mock_getattr = self.patcher.start()

        # Setup the exchange class constructor
        self.mock_exchange_class = Mock()
        self.mock_exchange_class.return_value = self.mock_exchange
        self.mock_getattr.return_value = self.mock_exchange_class

        # For rate limiter tests, add __name__ to mocked functions
        self.mock_exchange.fetch_markets = Mock(return_value=[])
        self.mock_exchange.fetch_markets.__name__ = 'fetch_markets'

        self.mock_exchange.fetch_ticker = Mock(return_value={})
        self.mock_exchange.fetch_ticker.__name__ = 'fetch_ticker'

        self.mock_exchange.fetch_ohlcv = Mock(return_value=[])
        self.mock_exchange.fetch_ohlcv.__name__ = 'fetch_ohlcv'

        self.mock_exchange.fetch_order_book = Mock(return_value={})
        self.mock_exchange.fetch_order_book.__name__ = 'fetch_order_book'

        # Patch time.time for all tests to avoid flakiness
        self.time_patcher = patch('time.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.return_value = 1000.0

        # Patch time.sleep for all tests to avoid waiting
        self.sleep_patcher = patch('time.sleep')
        self.mock_sleep = self.sleep_patcher.start()

    def teardown_method(self):
        """Tear down test fixtures."""
        self.patcher.stop()
        self.exchanges_patcher.stop()
        self.time_patcher.stop()
        self.sleep_patcher.stop()

    def test_initialization(self):
        """Test exchange service initialization."""
        # Test with minimum required parameters
        service = ExchangeService(exchange_id='binance')
        assert service.exchange_id == 'binance'
        assert service.api_key is None
        assert service.secret is None
        assert service.sandbox is False

        # Test with authentication
        service = ExchangeService(
            exchange_id='binance',
            api_key='test_key',
            secret='test_secret',
            password='test_password'
        )
        assert service.api_key == 'test_key'
        assert service.secret == 'test_secret'
        assert service.password == 'test_password'

    def test_initialize_exchange(self):
        """Test exchange initialization."""
        # Test with authentication
        ExchangeService(
            exchange_id='binance',
            api_key='test_key',
            secret='test_secret',
            timeout=5000
        )

        # Verify correct configuration was passed
        self.mock_exchange_class.assert_called_once()
        # Check if the first argument is a dictionary with expected keys
        call_args = self.mock_exchange_class.call_args[0][0]

        assert 'apiKey' in call_args
        assert call_args['apiKey'] == 'test_key'
        assert call_args['secret'] == 'test_secret'
        assert call_args['timeout'] == 5000
        assert call_args['enableRateLimit'] is True

    def test_initialize_exchange_sandbox(self):
        """Test exchange initialization with sandbox mode."""
        # Configure mock exchange to have sandbox URL
        self.mock_exchange.urls = {'test': 'https://testnet.binance.com'}
        self.mock_exchange.set_sandbox_mode = Mock()

        # Create service with sandbox=True
        ExchangeService(exchange_id='binance', sandbox=True)

        # Verify sandbox mode was set
        self.mock_exchange.set_sandbox_mode.assert_called_once_with(True)

    def test_get_exchange(self):
        """Test get_exchange method."""
        service = ExchangeService(exchange_id='binance')
        exchange = service.get_exchange()
        assert exchange == self.mock_exchange

    def test_invalid_exchange(self):
        """Test initialization with invalid exchange ID."""
        # Temporarily modify the mock to simulate an invalid exchange
        self.exchanges_patcher.stop()
        with patch('app.services.exchange_service.ccxt.exchanges', ['kraken', 'coinbase']):
            with pytest.raises(ValueError) as excinfo:
                ExchangeService(exchange_id='nonexistent')
            assert "not supported by CCXT" in str(excinfo.value)

        # Restart the original patcher
        self.exchanges_patcher = patch('app.services.exchange_service.ccxt.exchanges',
                                       ['binance', 'coinbase', 'kraken'])
        self.mock_exchanges = self.exchanges_patcher.start()

    def test_fetch_markets_caching(self):
        """Test fetch_markets with caching."""
        # Setup
        service = ExchangeService(exchange_id='binance', cache_ttl=60)

        # Mock the exchange's fetch_markets method
        mock_markets = [{'id': 'btcusdt', 'symbol': 'BTC/USDT'}]
        self.mock_exchange.fetch_markets = Mock(return_value=mock_markets)
        self.mock_exchange.fetch_markets.__name__ = 'fetch_markets'

        # First call should hit the API
        self.mock_time.return_value = 1000.0
        result1 = service.fetch_markets()
        assert result1 == mock_markets
        assert self.mock_exchange.fetch_markets.call_count == 1

        # Second call within TTL should use cache
        result2 = service.fetch_markets()
        assert result2 == mock_markets
        assert self.mock_exchange.fetch_markets.call_count == 1

        # Call after TTL should hit API again
        self.mock_time.return_value = 1061.0
        result3 = service.fetch_markets()
        assert result3 == mock_markets
        assert self.mock_exchange.fetch_markets.call_count == 2

        # Force refresh should always hit API
        result4 = service.fetch_markets(force_refresh=True)
        assert result4 == mock_markets
        assert self.mock_exchange.fetch_markets.call_count == 3

    def test_get_ticker_caching(self):
        """Test get_ticker with caching."""
        # Setup
        service = ExchangeService(exchange_id='binance', cache_ttl=60)

        # Mock the exchange's fetch_ticker method
        mock_ticker = {'symbol': 'BTC/USDT', 'last': 50000.0, 'bid': 49900.0, 'ask': 50100.0}
        self.mock_exchange.fetch_ticker = Mock(return_value=mock_ticker)
        self.mock_exchange.fetch_ticker.__name__ = 'fetch_ticker'

        # First call should hit the API - patched time is automatically 1000.0
        symbol = 'BTC/USDT'
        result1 = service.get_ticker(symbol)
        assert result1 == mock_ticker
        self.mock_exchange.fetch_ticker.assert_called_once_with(symbol, params={})

        # Second call within TTL should use cache - still at time 1000.0
        result2 = service.get_ticker(symbol)
        assert result2 == mock_ticker
        assert self.mock_exchange.fetch_ticker.call_count == 1

        # Call after TTL should hit API again
        self.mock_time.return_value = 1061.0
        result3 = service.get_ticker(symbol)
        assert result3 == mock_ticker
        assert self.mock_exchange.fetch_ticker.call_count == 2

        # Different symbol should hit API
        self.mock_exchange.fetch_ticker = Mock(return_value={'symbol': 'ETH/USDT', 'last': 3000.0})
        self.mock_exchange.fetch_ticker.__name__ = 'fetch_ticker'
        service.get_ticker('ETH/USDT')
        assert self.mock_exchange.fetch_ticker.call_count == 1

    def test_fetch_ohlcv(self):
        """Test fetch_ohlcv method."""
        # Setup
        service = ExchangeService(exchange_id='binance')

        # Mock the exchange's fetch_ohlcv method
        mock_ohlcv = [
            [1609459200000, 29000.0, 29100.0, 28900.0, 29050.0, 100.0],
            [1609545600000, 29050.0, 29200.0, 28800.0, 29150.0, 200.0]
        ]
        self.mock_exchange.fetch_ohlcv = Mock(return_value=mock_ohlcv)
        self.mock_exchange.fetch_ohlcv.__name__ = 'fetch_ohlcv'

        # Call the method
        symbol = 'BTC/USDT'
        timeframe = '1d'
        since = 1609459200000
        limit = 10
        result = service.fetch_ohlcv(symbol, timeframe, since, limit)

        # Verify result and call
        assert result == mock_ohlcv
        self.mock_exchange.fetch_ohlcv.assert_called_once_with(
            symbol, timeframe, since, limit, params={}
        )

    def test_fetch_ohlcv_unsupported_exchange(self):
        """Test fetch_ohlcv with an exchange that doesn't support it."""
        # Setup - exchange without OHLCV support
        self.mock_exchange.has['fetchOHLCV'] = False
        service = ExchangeService(exchange_id='binance')

        # Call should raise NotImplementedError
        with pytest.raises(NotImplementedError) as excinfo:
            service.fetch_ohlcv('BTC/USDT')
        assert "does not support fetching OHLCV data" in str(excinfo.value)

    def test_fetch_ohlcv_unsupported_timeframe(self):
        """Test fetch_ohlcv with an unsupported timeframe."""
        # Setup
        service = ExchangeService(exchange_id='binance')

        # Call with unsupported timeframe
        with pytest.raises(ValueError) as excinfo:
            service.fetch_ohlcv('BTC/USDT', timeframe='2d')
        assert "Timeframe '2d' not supported" in str(excinfo.value)

    def test_get_order_book(self):
        """Test get_order_book method."""
        # Setup
        service = ExchangeService(exchange_id='binance')

        # Mock the exchange's fetch_order_book method
        mock_order_book = {
            'bids': [[29000.0, 1.5], [28900.0, 2.0]],
            'asks': [[29100.0, 1.0], [29200.0, 3.0]],
            'timestamp': 1609459200000,
            'datetime': '2021-01-01T00:00:00Z'
        }
        self.mock_exchange.fetch_order_book = Mock(return_value=mock_order_book)
        self.mock_exchange.fetch_order_book.__name__ = 'fetch_order_book'

        # Call the method
        symbol = 'BTC/USDT'
        limit = 5
        result = service.get_order_book(symbol, limit)

        # Verify result and call
        assert result == mock_order_book
        self.mock_exchange.fetch_order_book.assert_called_once_with(
            symbol, limit, params={}
        )

    def test_get_order_book_unsupported_exchange(self):
        """Test get_order_book with an exchange that doesn't support it."""
        # Setup - exchange without order book support
        self.mock_exchange.has['fetchOrderBook'] = False
        service = ExchangeService(exchange_id='binance')

        # Call should raise NotImplementedError
        with pytest.raises(NotImplementedError) as excinfo:
            service.get_order_book('BTC/USDT')
        assert "does not support fetching order book" in str(excinfo.value)

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Directly test the rate limiting functionality
        with patch('app.services.exchange_rate_limiter.time.sleep') as mock_sleep:
            rate_limiter = RateLimiter(calls_per_second=0.5)  # 1 call per 2 seconds

            # Mock time.time to return controlled values
            with patch('app.services.exchange_rate_limiter.time.time') as mock_time:
                # First call at t=1000
                mock_time.return_value = 1000.0
                rate_limiter._enforce_rate_limit('test_method')

                # Second call at t=1001 (1s elapsed, should sleep for 1s)
                mock_time.return_value = 1001.0
                rate_limiter._enforce_rate_limit('test_method')

                # Check that sleep was called with expected value
                mock_sleep.assert_called_once()
                # We expect sleep to be called with a value close to 1.0
                sleep_time = mock_sleep.call_args[0][0]
                assert sleep_time > 0.9

    def test_rate_limit_backoff(self):
        """Test backoff on rate limit error."""
        # Setup
        # Directly test the backoff functionality to avoid mocking issues
        with patch('app.services.exchange_rate_limiter.time.sleep') as mock_sleep:
            # Create limiter with fixed parameters for testing
            rate_limiter = RateLimiter(
                max_retries=1,
                initial_backoff=0.1,
                jitter=False,
                calls_per_second=1000.0  # High value to minimize rate limiting
            )

            # Create a function that will fail once then succeed
            call_count = [0]

            def test_func():
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RateLimitExceeded("Rate limit exceeded")
                return "success"

            # Apply the decorator
            decorated_func = rate_limiter.with_backoff(test_func, (RateLimitExceeded,))

            # Call the function - should succeed after one retry
            result = decorated_func()

            # Verify results
            assert result == "success"
            assert call_count[0] == 2  # Initial call + 1 retry
            # Sleep is called once for rate limiting on initial call and again for the backoff
            assert mock_sleep.call_count >= 1  # At least sleep for the retry backoff

    def test_max_retries_exceeded(self):
        """Test exceeding maximum retries."""
        # Directly test the backoff functionality to avoid mocking issues
        with patch('app.services.exchange_rate_limiter.time.sleep') as mock_sleep:
            # Create limiter with fixed parameters for testing
            rate_limiter = RateLimiter(
                max_retries=2,
                initial_backoff=0.1,
                jitter=False,
                calls_per_second=1000.0  # High value to minimize rate limiting
            )

            # Create a function that always fails
            def failing_func():
                raise RateLimitExceeded("Rate limit exceeded")

            # Apply the decorator
            decorated_func = rate_limiter.with_backoff(failing_func, (RateLimitExceeded,))

            # Call the function - should raise exception after max retries
            with pytest.raises(RateLimitExceeded):
                decorated_func()

            # Verify results
            # Each retry can trigger 2 sleep calls: one for rate limiting, one for backoff
            # For 2 retries, we expect at least 2 sleep calls (backoffs) and up to 4 total
            assert mock_sleep.call_count >= 2  # At least the backoff sleeps

    def test_with_backoff_decorator(self):
        """Test with_backoff decorator."""
        # Directly test with patches to avoid multiple sleep calls
        with patch('app.services.exchange_rate_limiter.time.sleep') as mock_sleep:
            # Setup
            limiter = RateLimiter(
                max_retries=2,
                initial_backoff=0.1,
                jitter=False,
                calls_per_second=1000.0  # High value to minimize rate limiting
            )

            # Create a function that fails twice then succeeds
            call_count = [0]

            def test_func():
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise RateLimitExceeded("Rate limit exceeded")
                return "success"

            # Apply the decorator
            decorated_func = limiter.with_backoff(test_func, (RateLimitExceeded,))

            # Call the function - should succeed after two retries
            result = decorated_func()

            # Verify results
            assert result == "success"
            assert call_count[0] == 3  # Initial call + 2 retries
            # Each retry can trigger sleep calls from both rate limiting and backoff
            # Expect at least 2 backoff sleeps
            assert mock_sleep.call_count >= 2  # At least the backoff sleeps

    def test_with_backoff_max_retries(self):
        """Test with_backoff with max retries exceeded."""
        # Directly test with patches to avoid multiple sleep calls
        with patch('app.services.exchange_rate_limiter.time.sleep') as mock_sleep:
            limiter = RateLimiter(
                max_retries=2,
                initial_backoff=0.1,
                jitter=False,
                calls_per_second=1000.0  # High value to minimize rate limiting
            )

            # Create a function that always fails
            def failing_func():
                raise RateLimitExceeded("Rate limit exceeded")

            # Apply the decorator
            decorated_func = limiter.with_backoff(failing_func, (RateLimitExceeded,))

            # Call the function - should raise exception after max retries
            with pytest.raises(RateLimitExceeded):
                decorated_func()

            # Verify results
            # Each retry includes both rate limiting and backoff sleeps
            # For max 2 retries, expect at least 2 sleep calls
            assert mock_sleep.call_count >= 2  # At least the backoff sleeps

    def test_close(self):
        """Test close method."""
        # Setup with mock session
        self.mock_exchange.session = Mock()
        self.mock_exchange.session.close = Mock()

        service = ExchangeService(exchange_id='binance')

        # Populate caches
        service._markets_cache = {'mock': 'data'}
        service._markets_cache_timestamp = 12345.0
        service._tickers_cache = {'BTC/USDT': {'mock': 'ticker'}}
        service._tickers_cache_timestamp = {'BTC/USDT': 12345.0}

        # Call close
        service.close()

        # Verify caches are cleared
        assert service._markets_cache == {}
        assert service._markets_cache_timestamp is None
        assert service._tickers_cache == {}
        assert service._tickers_cache_timestamp == {}

        # Verify session is closed
        self.mock_exchange.session.close.assert_called_once()

    def test_integration_with_rate_limiter(self):
        """Test that the ExchangeService correctly uses the RateLimiter."""
        # Setup
        service = ExchangeService(
            exchange_id='binance',
            rate_limit_calls_per_second=0.5,  # 1 call per 2 seconds
            max_retries=2
        )

        # Create a mock for the service's rate_limiter
        mock_rate_limiter = MagicMock()
        service.rate_limiter = mock_rate_limiter

        # Configure the with_backoff decorator to track calls
        def fake_with_backoff(func, errors=None):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        mock_rate_limiter.with_backoff.side_effect = fake_with_backoff

        # Call some service methods
        service.get_ticker('BTC/USDT')
        service.fetch_markets()
        service.fetch_ohlcv('BTC/USDT', '1h')

        # Verify that with_backoff was used
        assert mock_rate_limiter.with_backoff.call_count >= 3  # Should be called for each method


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(calls_per_second=2.0, max_retries=3)
        assert limiter.calls_per_second == 2.0
        assert limiter.min_interval == 0.5
        assert limiter.max_retries == 3

    def test_enforce_rate_limit(self):
        """Test rate limit enforcement."""
        # Setup
        limiter = RateLimiter(calls_per_second=1.0)  # 1 call per second

        # Patch time.time
        with patch('time.time') as mock_time:
            # First call at t=1000
            mock_time.return_value = 1000.0
            limiter._enforce_rate_limit('test_method')

            # Second call at t=1000.5 (0.5s elapsed)
            mock_time.return_value = 1000.5

            # Patch sleep
            with patch('time.sleep') as mock_sleep:
                limiter._enforce_rate_limit('test_method')
                # Should sleep for 0.5s (1.0s interval - 0.5s elapsed)
                mock_sleep.assert_called_once_with(0.5)

    def test_rate_limited_decorator(self):
        """Test rate_limited decorator."""
        # Setup
        limiter = RateLimiter(calls_per_second=0.5)  # 1 call per 2 seconds

        # Create a test function with a name
        def test_func():
            return "test"

        # Apply the decorator
        decorated_func = limiter.rate_limited(test_func)

        # Patch the _enforce_rate_limit method to verify it's called
        with patch.object(limiter, '_enforce_rate_limit') as mock_enforce:
            # First call
            assert decorated_func() == "test"
            mock_enforce.assert_called_once_with('test_func')

            # Second call
            mock_enforce.reset_mock()
            assert decorated_func() == "test"
            mock_enforce.assert_called_once_with('test_func')

    def test_calculate_backoff(self):
        """Test backoff calculation."""
        # Setup without jitter for deterministic testing
        limiter = RateLimiter(initial_backoff=1.0, backoff_factor=2.0, jitter=False)

        # Verify exponential backoff
        assert limiter._calculate_backoff(0) == 1.0
        assert limiter._calculate_backoff(1) == 2.0
        assert limiter._calculate_backoff(2) == 4.0
        assert limiter._calculate_backoff(3) == 8.0
