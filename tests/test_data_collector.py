"""
Tests for the DataCollector service.

This module contains tests for the DataCollector service, which is responsible for
collecting various types of market data, managing data storage and organization,
and handling both historical and real-time data.
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from app.services.data_collector import DataCollector
from app.core.exceptions import ValidationError, APIError, ExchangeConnectionError


@pytest.fixture
def mock_exchange_service():
    """Fixture for a mock exchange service."""
    mock = Mock()

    # Mock OHLCV data (timestamp, open, high, low, close, volume)
    mock.fetch_ohlcv.return_value = [
        [1609459200000, 29000.0, 29100.0, 28900.0, 29050.0, 100.5],
        [1609462800000, 29050.0, 29200.0, 29000.0, 29150.0, 105.2],
        [1609466400000, 29150.0, 29300.0, 29100.0, 29250.0, 98.7],
    ]

    mock.fetch_historical_ohlcv.return_value = mock.fetch_ohlcv.return_value

    # Mock order book data
    mock.fetch_order_book_snapshot.return_value = {
        "bids": [[28900.0, 1.5], [28850.0, 2.3], [28800.0, 3.1]],
        "asks": [[29100.0, 1.2], [29150.0, 2.0], [29200.0, 2.7]],
    }

    # Mock ticker data
    mock.get_ticker.return_value = {
        "last": 29050.0,
        "bid": 29000.0,
        "ask": 29100.0,
        "volume": 1000.5,
        "timestamp": 1609459200000,
    }

    # Mock recent trades
    mock.fetch_recent_trades.return_value = [
        {"id": "1", "price": 29050.0, "amount": 0.5, "side": "buy", "timestamp": 1609459100000},
        {"id": "2", "price": 29075.0, "amount": 0.3, "side": "sell", "timestamp": 1609459150000},
        {"id": "3", "price": 29025.0, "amount": 0.8, "side": "buy", "timestamp": 1609459190000},
    ]

    return mock


@pytest.fixture
def mock_data_storage():
    """Fixture for a mock data storage service."""
    mock = Mock()

    # Mock storage methods
    mock.store_ohlcv.return_value = 3  # Number of records stored
    mock.store_order_book.return_value = 1  # ID of stored order book
    mock.store_market_snapshot.return_value = 1  # ID of stored snapshot

    # Mock continuity check
    mock.check_data_continuity.return_value = [
        (datetime.utcnow() - timedelta(hours=5), datetime.utcnow() - timedelta(hours=3))
    ]

    return mock


@pytest.fixture
def mock_session_provider():
    """Fixture for a mock session provider function."""
    mock_session = MagicMock()

    def get_session():
        return mock_session

    return get_session


@pytest.fixture
def data_collector(mock_exchange_service, mock_data_storage, mock_session_provider):
    """Fixture for a DataCollector instance with mock dependencies."""
    return DataCollector(
        exchange_service=mock_exchange_service,
        data_storage=mock_data_storage,
        session_provider=mock_session_provider,
        logger=logging.getLogger("test_logger"),
    )


class TestDataCollector:
    """Tests for the DataCollector class."""

    def test_init(self, data_collector):
        """Test initialization of DataCollector."""
        assert data_collector._exchange_service is not None
        assert data_collector._data_storage is not None
        assert data_collector._session_provider is not None
        assert data_collector._max_retries == 3
        assert data_collector._retry_delay == 1.0
        assert data_collector._backoff_factor == 2.0
        assert data_collector._max_concurrent_requests == 5
        assert data_collector._collection_stats == {}

    def test_collect_ohlcv_success(self, data_collector, mock_exchange_service, mock_data_storage):
        """Test successful OHLCV data collection."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframes = ["1h"]

        # Act
        result = data_collector.collect_ohlcv(symbols, timeframes)

        # Assert
        mock_exchange_service.fetch_ohlcv.assert_called_once_with(
            "BTC/USD", "1h", None
        )
        mock_data_storage.store_ohlcv.assert_called_once()

        assert result["BTC/USD"]["1h"]["status"] == "success"
        assert result["BTC/USD"]["1h"]["count"] == 3

    def test_collect_ohlcv_with_time_range(self, data_collector, mock_exchange_service):
        """Test OHLCV collection with time range."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframes = ["1h"]
        start_time = datetime(2021, 1, 1)
        end_time = datetime(2021, 1, 2)

        # Act
        result = data_collector.collect_ohlcv(symbols, timeframes, start_time, end_time)

        # Assert
        mock_exchange_service.fetch_historical_ohlcv.assert_called_once_with(
            "BTC/USD", "1h", int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000)
        )

        assert result["BTC/USD"]["1h"]["status"] == "success"

    def test_collect_ohlcv_no_data(self, data_collector, mock_exchange_service):
        """Test OHLCV collection when no data is available."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframes = ["1h"]
        mock_exchange_service.fetch_ohlcv.return_value = []

        # Act
        result = data_collector.collect_ohlcv(symbols, timeframes)

        # Assert
        assert result["BTC/USD"]["1h"]["status"] == "no_data"
        assert result["BTC/USD"]["1h"]["count"] == 0

    def test_collect_ohlcv_validation_failure(self, data_collector, mock_exchange_service):
        """Test OHLCV collection with validation failure."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframes = ["1h"]

        # Invalid OHLCV data (low > high)
        mock_exchange_service.fetch_ohlcv.return_value = [
            [1609459200000, 29000.0, 28800.0, 29100.0, 29050.0, 100.5],  # high < low!
        ]

        # Act
        result = data_collector.collect_ohlcv(symbols, timeframes)

        # Assert
        assert result["BTC/USD"]["1h"]["status"] == "validation_error"
        assert "error" in result["BTC/USD"]["1h"]

    def test_collect_ohlcv_api_error(self, data_collector, mock_exchange_service):
        """Test OHLCV collection handling API errors."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframes = ["1h"]
        mock_exchange_service.fetch_ohlcv.side_effect = APIError("API error")

        # Act
        result = data_collector.collect_ohlcv(symbols, timeframes)

        # Assert
        assert result["BTC/USD"]["1h"]["status"] == "exchange_error"
        assert "API error" in result["BTC/USD"]["1h"]["error"]

    def test_collect_ohlcv_connection_error(self, data_collector, mock_exchange_service):
        """Test OHLCV collection handling connection errors."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframes = ["1h"]
        mock_exchange_service.fetch_ohlcv.side_effect = ExchangeConnectionError("Connection error")

        # Act
        result = data_collector.collect_ohlcv(symbols, timeframes)

        # Assert
        assert result["BTC/USD"]["1h"]["status"] == "exchange_error"
        assert "Connection error" in result["BTC/USD"]["1h"]["error"]

    def test_collect_ohlcv_without_storing(self, data_collector, mock_exchange_service, mock_data_storage):
        """Test OHLCV collection without storing data."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframes = ["1h"]

        # Act
        result = data_collector.collect_ohlcv(symbols, timeframes, store=False)

        # Assert
        mock_exchange_service.fetch_ohlcv.assert_called_once()
        # Storage should not be called
        mock_data_storage.store_ohlcv.assert_not_called()

        assert result["BTC/USD"]["1h"]["status"] == "success"
        assert "data" in result["BTC/USD"]["1h"]

    @pytest.mark.asyncio
    async def test_collect_ohlcv_async(self, data_collector):
        """Test asynchronous OHLCV collection."""
        # Arrange
        symbols = ["BTC/USD", "ETH/USD"]
        timeframes = ["1h", "4h"]

        # Mock the collect_ohlcv method to simulate synchronous collection
        original_collect_ohlcv = data_collector.collect_ohlcv

        def mock_collect_ohlcv(symbols, timeframes, *args, **kwargs):
            # Simplified mock that returns success for any symbol/timeframe combination
            return {symbols[0]: {timeframes[0]: {"status": "success", "count": 3}}}

        data_collector.collect_ohlcv = mock_collect_ohlcv

        try:
            # Act
            result = await data_collector.collect_ohlcv_async(symbols, timeframes)

            # Assert
            assert len(result) == 2  # Two symbols
            for symbol in symbols:
                assert symbol in result
                for timeframe in timeframes:
                    assert timeframe in result[symbol]

        finally:
            # Restore the original method
            data_collector.collect_ohlcv = original_collect_ohlcv

    def test_collect_order_book(self, data_collector, mock_exchange_service, mock_data_storage):
        """Test order book collection."""
        # Arrange
        symbols = ["BTC/USD"]
        depth = 10

        # Act
        result = data_collector.collect_order_book(symbols, depth)

        # Assert
        mock_exchange_service.fetch_order_book_snapshot.assert_called_once_with(
            "BTC/USD", depth=depth
        )
        mock_data_storage.store_order_book.assert_called_once()

        assert result["BTC/USD"]["status"] == "success"
        assert "timestamp" in result["BTC/USD"]

    def test_collect_order_book_with_retry(self, data_collector, mock_exchange_service):
        """Test order book collection with retry logic."""
        # Arrange
        symbols = ["BTC/USD"]

        # Mock fetch_order_book_snapshot to fail once then succeed
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIError("Temporary API error")
            return {
                "bids": [[28900.0, 1.5]],
                "asks": [[29100.0, 1.2]],
            }

        mock_exchange_service.fetch_order_book_snapshot.side_effect = side_effect

        # Act
        result = data_collector.collect_order_book(symbols)

        # Assert
        assert mock_exchange_service.fetch_order_book_snapshot.call_count == 2
        assert result["BTC/USD"]["status"] == "success"

    def test_collect_order_book_failure(self, data_collector, mock_exchange_service):
        """Test order book collection failure handling."""
        # Arrange
        symbols = ["BTC/USD"]

        # Make the API call fail consistently
        mock_exchange_service.fetch_order_book_snapshot.side_effect = APIError("Persistent API error")

        # Set a low retry count for the test
        data_collector._max_retries = 2

        # Act
        result = data_collector.collect_order_book(symbols)

        # Assert
        assert mock_exchange_service.fetch_order_book_snapshot.call_count == 3  # Initial + 2 retries
        assert result["BTC/USD"]["status"] == "exchange_error"
        assert "Persistent API error" in result["BTC/USD"]["error"]

    def test_collect_market_snapshots(self, data_collector, mock_exchange_service, mock_data_storage):
        """Test market snapshot collection."""
        # Arrange
        symbols = ["BTC/USD"]

        # Act
        result = data_collector.collect_market_snapshots(symbols)

        # Assert
        mock_exchange_service.get_ticker.assert_called_once_with("BTC/USD")
        mock_exchange_service.fetch_order_book_snapshot.assert_called_once_with("BTC/USD")
        mock_exchange_service.fetch_recent_trades.assert_called_once_with("BTC/USD")
        mock_data_storage.store_market_snapshot.assert_called_once()

        assert result["BTC/USD"]["status"] == "success"
        assert "timestamp" in result["BTC/USD"]

    def test_collect_market_snapshots_partial_failure(self, data_collector, mock_exchange_service, mock_data_storage):
        """Test market snapshot collection with some components failing."""
        # Arrange
        symbols = ["BTC/USD"]

        # Make ticker fetching fail
        mock_exchange_service.get_ticker.side_effect = APIError("Ticker API error")

        # Act
        result = data_collector.collect_market_snapshots(symbols)

        # Assert
        # Even with ticker failing, the snapshot should still be collected and stored
        mock_data_storage.store_market_snapshot.assert_called_once()
        assert result["BTC/USD"]["status"] == "success"

    def test_collect_market_snapshots_without_storing(self, data_collector, mock_exchange_service, mock_data_storage):
        """Test market snapshot collection without storing."""
        # Arrange
        symbols = ["BTC/USD"]

        # Act
        result = data_collector.collect_market_snapshots(symbols, store=False)

        # Assert
        mock_data_storage.store_market_snapshot.assert_not_called()
        assert result["BTC/USD"]["status"] == "success"
        assert "data" in result["BTC/USD"]

    def test_backfill_missing_data(self, data_collector, mock_exchange_service, mock_data_storage):
        """Test backfilling missing data."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframe = "1h"

        # Configure the mock to return a single gap
        gap_start = datetime.utcnow() - timedelta(hours=5)
        gap_end = datetime.utcnow() - timedelta(hours=3)
        mock_data_storage.check_data_continuity.return_value = [(gap_start, gap_end)]

        # Act
        result = data_collector.backfill_missing_data(symbols, timeframe)

        # Assert
        mock_data_storage.check_data_continuity.assert_called_once()
        mock_exchange_service.fetch_historical_ohlcv.assert_called_once_with(
            "BTC/USD", "1h", int(gap_start.timestamp() * 1000), int(gap_end.timestamp() * 1000)
        )
        mock_data_storage.store_ohlcv.assert_called_once()

        assert result["BTC/USD"]["status"] == "backfilled"
        assert result["BTC/USD"]["gaps_found"] == 1

    def test_backfill_missing_data_no_gaps(self, data_collector, mock_data_storage):
        """Test backfilling when no gaps are found."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframe = "1h"

        # Configure the mock to return no gaps
        mock_data_storage.check_data_continuity.return_value = []

        # Act
        result = data_collector.backfill_missing_data(symbols, timeframe)

        # Assert
        mock_data_storage.check_data_continuity.assert_called_once()
        # No historical data should be fetched
        assert result["BTC/USD"]["status"] == "no_gaps"
        assert result["BTC/USD"]["gaps_found"] == 0

    def test_backfill_missing_data_check_only(self, data_collector, mock_data_storage, mock_exchange_service):
        """Test backfilling in check-only mode."""
        # Arrange
        symbols = ["BTC/USD"]
        timeframe = "1h"

        # Configure the mock to return a single gap
        gap_start = datetime.utcnow() - timedelta(hours=5)
        gap_end = datetime.utcnow() - timedelta(hours=3)
        mock_data_storage.check_data_continuity.return_value = [(gap_start, gap_end)]

        # Act
        result = data_collector.backfill_missing_data(symbols, timeframe, check_only=True)

        # Assert
        mock_data_storage.check_data_continuity.assert_called_once()
        # Historical data should not be fetched in check-only mode
        mock_exchange_service.fetch_historical_ohlcv.assert_not_called()

        assert result["BTC/USD"]["status"] == "gaps_found"
        assert result["BTC/USD"]["gaps_found"] == 1
        assert "gap_periods" in result["BTC/USD"]

    def test_validate_collected_data_ohlcv(self, data_collector):
        """Test OHLCV data validation."""
        # Arrange
        valid_ohlcv = [
            [1609459200000, 29000.0, 29100.0, 28900.0, 29050.0, 100.5],
            [1609462800000, 29050.0, 29200.0, 29000.0, 29150.0, 105.2],
            [1609466400000, 29150.0, 29300.0, 29100.0, 29250.0, 98.7],
        ]

        # Act
        result = data_collector.validate_collected_data(valid_ohlcv)

        # Assert
        assert result is True

    def test_validate_collected_data_ohlcv_invalid(self, data_collector):
        """Test validation of invalid OHLCV data."""
        # Arrange
        # Invalid: low > high
        invalid_ohlcv = [
            [1609459200000, 29000.0, 28800.0, 29100.0, 29050.0, 100.5],
        ]

        # Act/Assert
        with pytest.raises(ValidationError) as excinfo:
            data_collector.validate_collected_data(invalid_ohlcv)

        assert "Low price" in str(excinfo.value)
        assert "greater than high price" in str(excinfo.value)

    def test_validate_collected_data_order_book(self, data_collector):
        """Test order book data validation."""
        # Arrange
        valid_order_book = {
            "bids": [[29000.0, 1.5], [28950.0, 2.0], [28900.0, 2.5]],
            "asks": [[29100.0, 1.2], [29150.0, 1.8], [29200.0, 2.2]],
        }

        # Act
        result = data_collector.validate_collected_data(valid_order_book)

        # Assert
        assert result is True

    def test_validate_collected_data_order_book_invalid(self, data_collector):
        """Test validation of invalid order book data."""
        # Arrange
        # Invalid: bids not in descending order
        invalid_order_book = {
            "bids": [[28900.0, 1.5], [29000.0, 2.0]],  # Wrong order
            "asks": [[29100.0, 1.2], [29150.0, 1.8]],
        }

        # Act/Assert
        with pytest.raises(ValidationError) as excinfo:
            data_collector.validate_collected_data(invalid_order_book)

        assert "Bids are not in descending order" in str(excinfo.value)

    def test_validate_collected_data_market_snapshot(self, data_collector):
        """Test market snapshot validation."""
        # Arrange
        valid_snapshot = {
            "timestamp": datetime.utcnow(),
            "ticker": {
                "last": 29050.0,
                "bid": 29000.0,
                "ask": 29100.0,
                "volume": 1000.5,
            },
            "order_book": {
                "bids": [[29000.0, 1.5], [28950.0, 2.0]],
                "asks": [[29100.0, 1.2], [29150.0, 1.8]],
            },
            "trades": [
                {"price": 29050.0, "amount": 0.5, "side": "buy"},
                {"price": 29075.0, "amount": 0.3, "side": "sell"},
            ],
        }

        # Act
        result = data_collector.validate_collected_data(valid_snapshot)

        # Assert
        assert result is True

    def test_validate_collected_data_market_snapshot_invalid(self, data_collector):
        """Test validation of invalid market snapshot."""
        # Arrange
        # Invalid: missing required ticker field
        invalid_snapshot = {
            "timestamp": datetime.utcnow(),
            "ticker": {
                "last": 29050.0,
                "bid": 29000.0,
                # Missing "ask"
                "volume": 1000.5,
            },
        }

        # Act/Assert
        with pytest.raises(ValidationError) as excinfo:
            data_collector.validate_collected_data(invalid_snapshot)

        assert "Ticker missing required field" in str(excinfo.value)

    def test_validate_collected_data_unknown_format(self, data_collector):
        """Test validation of unknown data format."""
        # Arrange
        unknown_data = {"foo": "bar"}  # Doesn't match any known format

        # Act/Assert
        with pytest.raises(ValidationError) as excinfo:
            data_collector.validate_collected_data(unknown_data)

        assert "Unknown data format for validation" in str(excinfo.value)

    def test_update_collection_stats(self, data_collector):
        """Test collection statistics tracking."""
        # Arrange
        symbol = "BTC/USD"
        data_type = "ohlcv"
        timeframe = "1h"

        # Act - Track a successful collection
        data_collector._update_collection_stats(symbol, data_type, timeframe, True)

        # Act - Track a failed collection
        data_collector._update_collection_stats(
            symbol, data_type, timeframe, False, "API error"
        )

        # Assert
        stats = data_collector.get_collection_stats()
        key = f"{symbol}:{data_type}:{timeframe}"

        assert key in stats
        assert stats[key]["total_attempts"] == 2
        assert stats[key]["successful_attempts"] == 1
        assert stats[key]["error_count"] == 1
        assert stats[key]["last_error"] == "API error"

    def test_get_collection_stats_filtered(self, data_collector):
        """Test filtering collection statistics."""
        # Arrange
        # Add some stats for different symbols and timeframes
        data_collector._update_collection_stats("BTC/USD", "ohlcv", "1h", True)
        data_collector._update_collection_stats("BTC/USD", "ohlcv", "4h", True)
        data_collector._update_collection_stats("ETH/USD", "ohlcv", "1h", True)
        data_collector._update_collection_stats("BTC/USD", "order_book", None, True)

        # Act - Filter by symbol
        btc_stats = data_collector.get_collection_stats(symbol="BTC/USD")

        # Assert
        assert len(btc_stats) == 3  # 3 BTC records
        for key in btc_stats:
            assert "BTC/USD" in key

        # Act - Filter by data type
        ohlcv_stats = data_collector.get_collection_stats(data_type="ohlcv")

        # Assert
        assert len(ohlcv_stats) == 3  # 3 OHLCV records
        for key in ohlcv_stats:
            assert "ohlcv" in key

        # Act - Filter by timeframe
        hourly_stats = data_collector.get_collection_stats(timeframe="1h")

        # Assert
        assert len(hourly_stats) == 2  # 2 records with 1h timeframe
        for key in hourly_stats:
            assert ":1h" in key

    def test_reset_collection_stats(self, data_collector):
        """Test resetting collection statistics."""
        # Arrange
        data_collector._update_collection_stats("BTC/USD", "ohlcv", "1h", True)
        data_collector._update_collection_stats("ETH/USD", "ohlcv", "1h", True)

        # Verify stats exist
        assert len(data_collector.get_collection_stats()) == 2

        # Act
        data_collector.reset_collection_stats()

        # Assert
        assert len(data_collector.get_collection_stats()) == 0

    def test_export_collection_stats_as_dataframe(self, data_collector):
        """Test exporting collection statistics as DataFrame."""
        # Arrange
        data_collector._update_collection_stats("BTC/USD", "ohlcv", "1h", True)
        data_collector._update_collection_stats("ETH/USD", "ohlcv", "1h", False, "Error")

        # Act
        df = data_collector.export_collection_stats()

        # Assert
        assert len(df) == 2  # 2 rows for 2 stats entries
        assert "symbol" in df.columns
        assert "data_type" in df.columns
        assert "timeframe" in df.columns
        assert "total_attempts" in df.columns
        assert "successful_attempts" in df.columns

        # Verify data
        btc_row = df[df["symbol"] == "BTC/USD"].iloc[0]
        assert btc_row["successful_attempts"] == 1
        assert btc_row["error_count"] == 0

        eth_row = df[df["symbol"] == "ETH/USD"].iloc[0]
        assert eth_row["successful_attempts"] == 0
        assert eth_row["error_count"] == 1
        assert eth_row["last_error"] == "Error"
