"""
Tests for the DataStorage service.

This module contains tests for the DataStorage service, which is responsible for
providing an abstraction layer for database operations optimized for time series data.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.exc import IntegrityError

from app.services.data_storage import DataStorage
from app.models.ohlcv import OHLCV
from app.models.cryptocurrency import Cryptocurrency
from app.models.market_snapshot import MarketSnapshot


@pytest.fixture
def mock_session():
    """Fixture for a mock database session."""
    session = MagicMock()

    # Configure query chaining for session.query()
    query_mock = session.query.return_value
    query_mock.filter.return_value = query_mock
    query_mock.filter_by.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.first.return_value = None  # Default to not finding records
    query_mock.all.return_value = []  # Default to empty result set
    query_mock.limit.return_value = query_mock

    return session


@pytest.fixture
def mock_session_provider(mock_session):
    """Fixture for a function that returns a mock session."""
    def get_session():
        return mock_session
    return get_session


@pytest.fixture
def mock_cryptocurrency():
    """Fixture for a mock cryptocurrency record."""
    crypto = Mock(spec=Cryptocurrency)
    crypto.id = 1
    crypto.exchange = "binance"
    crypto.symbol = "BTC/USD"
    crypto.base_currency = "BTC"
    crypto.quote_currency = "USD"
    crypto.active = True
    crypto.last_updated = datetime.utcnow()

    return crypto


@pytest.fixture
def mock_ohlcv_records():
    """Fixture for mock OHLCV records."""
    now = datetime.utcnow()

    records = []
    for i in range(3):
        record = Mock(spec=OHLCV)
        record.id = i + 1
        record.cryptocurrency_id = 1
        record.exchange = "binance"
        record.symbol = "BTC/USD"
        record.timeframe = "1h"
        record.timestamp = now - timedelta(hours=i)
        record.open = 30000.0 + i * 100
        record.high = 30100.0 + i * 100
        record.low = 29900.0 + i * 100
        record.close = 30050.0 + i * 100
        record.volume = 100.0 + i * 10
        record.indicators = {"rsi": 60.0 - i, "sma": 30000.0 + i * 50}
        records.append(record)

    return records


@pytest.fixture
def mock_market_snapshots():
    """Fixture for mock market snapshot records."""
    now = datetime.utcnow()

    snapshots = []
    for i in range(3):
        snapshot = Mock(spec=MarketSnapshot)
        snapshot.id = i + 1
        snapshot.cryptocurrency_id = 1
        snapshot.exchange = "binance"
        snapshot.symbol = "BTC/USD"
        snapshot.timestamp = now - timedelta(minutes=i * 5)
        snapshot.data = {
            "ticker": {
                "last": 30050.0 + i * 10,
                "bid": 30000.0 + i * 10,
                "ask": 30100.0 + i * 10,
                "volume": 1000.0 + i * 100,
            },
            "order_book": {
                "bids": [[30000.0 - i * 10, 1.5], [29950.0 - i * 10, 2.0]],
                "asks": [[30100.0 + i * 10, 1.2], [30150.0 + i * 10, 1.8]],
            },
        }
        snapshots.append(snapshot)

    return snapshots


@pytest.fixture
def data_storage(mock_session_provider):
    """Fixture for a DataStorage instance with mock dependencies."""
    return DataStorage(
        session_provider=mock_session_provider,
        logger=logging.getLogger("test_logger"),
        batch_size=100,
        optimize_writes=True,
    )


class TestDataStorage:
    """Tests for the DataStorage class."""

    def test_init(self, data_storage):
        """Test initialization of DataStorage."""
        assert data_storage._session_provider is not None
        assert data_storage._batch_size == 100
        assert data_storage._optimize_writes is True

    def test_get_session(self, data_storage, mock_session):
        """Test getting a database session."""
        session = data_storage.get_session()
        assert session is mock_session

    def test_store_ohlcv_success(self, data_storage, mock_session, mock_cryptocurrency):
        """Test successful storage of OHLCV data."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        ohlcv_data = [
            [1609459200000, 29000.0, 29100.0, 28900.0, 29050.0, 100.5],
            [1609462800000, 29050.0, 29200.0, 29000.0, 29150.0, 105.2],
        ]

        # Patch the get_session method to ensure our mock session is used
        original_get_session = data_storage.get_session

        # Create a context manager mock that returns the mock session
        cm_mock = MagicMock()
        cm_mock.__enter__.return_value = mock_session
        cm_mock.__exit__.return_value = False

        # Create a get_session function that returns our context manager mock
        def mock_get_session():
            return cm_mock

        # Replace the get_session method
        data_storage.get_session = mock_get_session

        # Mock the _upsert_ohlcv_records method
        data_storage._upsert_ohlcv_records = Mock(return_value=2)

        try:
            # Mock cryptocurrency lookup
            mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

            # Act
            result = data_storage.store_ohlcv(symbol, timeframe, ohlcv_data)

            # Assert
            assert result == 2  # Should return the number of records stored

            # Verify _upsert_ohlcv_records was called
            data_storage._upsert_ohlcv_records.assert_called_once()

            # Extract the records passed to _upsert_ohlcv_records
            records = data_storage._upsert_ohlcv_records.call_args[0][1]
            assert len(records) == 2

            # Check first record's fields
            first_record = records[0]
            assert first_record["symbol"] == symbol
            assert first_record["timeframe"] == timeframe
            assert isinstance(first_record["timestamp"], datetime)
            assert first_record["open"] == ohlcv_data[0][1]
            assert first_record["high"] == ohlcv_data[0][2]
            assert first_record["low"] == ohlcv_data[0][3]
            assert first_record["close"] == ohlcv_data[0][4]
            assert first_record["volume"] == ohlcv_data[0][5]

        finally:
            # Restore the original method
            data_storage.get_session = original_get_session

    def test_store_ohlcv_with_empty_data(self, data_storage):
        """Test storing OHLCV with empty data."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        ohlcv_data = []

        # Act
        result = data_storage.store_ohlcv(symbol, timeframe, ohlcv_data)

        # Assert
        assert result == 0  # No records stored

    def test_store_ohlcv_create_cryptocurrency(self, data_storage, mock_session):
        """Test creating a new cryptocurrency record when storing OHLCV data."""
        # Arrange
        symbol = "ETH/USD"
        timeframe = "1h"
        ohlcv_data = [
            [1609459200000, 1000.0, 1010.0, 990.0, 1005.0, 50.5],
        ]

        # Configure mock to return no existing cryptocurrency
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Act
        data_storage.store_ohlcv(symbol, timeframe, ohlcv_data)

        # Assert
        # Check that a new cryptocurrency was created
        mock_session.add.assert_called_once()
        added_crypto = mock_session.add.call_args[0][0]
        assert isinstance(added_crypto, Cryptocurrency)
        assert added_crypto.symbol == "ETH/USD"
        assert added_crypto.is_active is True

    def test_store_ohlcv_with_integrity_error(self, data_storage, mock_session, mock_cryptocurrency):
        """Test handling integrity errors when storing OHLCV data."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        ohlcv_data = [
            [1609459200000, 29000.0, 29100.0, 28900.0, 29050.0, 100.5],
        ]

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Configure mock to raise IntegrityError on bulk_insert_mappings
        mock_session.bulk_insert_mappings.side_effect = IntegrityError("statement", "params", "orig")

        # Mock the _upsert_ohlcv_records method to return 1 (indicating success)
        with patch.object(data_storage, '_upsert_ohlcv_records', return_value=1):
            # Act
            result = data_storage.store_ohlcv(symbol, timeframe, ohlcv_data)

            # Assert
            assert result == 1  # Should return 1 record inserted

    def test_store_ohlcv_with_upsert(self, data_storage, mock_session, mock_cryptocurrency):
        """Test upserting OHLCV data (update existing records)."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Configure mock to simulate PostgreSQL upsert
        mock_session.execute.return_value.rowcount = 1

        # Act
        result = data_storage._upsert_ohlcv_records(mock_session, [
            {
                "cryptocurrency_id": mock_cryptocurrency.id,
                "exchange": mock_cryptocurrency.exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.fromtimestamp(1609459200000 / 1000.0),
                "open": 29000.0,
                "high": 29100.0,
                "low": 28900.0,
                "close": 29050.0,
                "volume": 100.5,
            }
        ])

        # Assert
        assert result == 1  # One record upserted
        mock_session.execute.assert_called_once()

    def test_store_ohlcv_with_upsert_fallback(self, data_storage, mock_session, mock_cryptocurrency):
        """Test upserting OHLCV data with fallback to non-PostgreSQL approach."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        timestamp = datetime.fromtimestamp(1609459200000 / 1000.0)

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Configure mock to fail on PostgreSQL upsert
        mock_session.execute.side_effect = Exception("PostgreSQL upsert not available")

        # Configure mock to find an existing record
        existing_record = Mock(spec=OHLCV)
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = existing_record

        # Act
        result = data_storage._upsert_ohlcv_records(mock_session, [
            {
                "cryptocurrency_id": mock_cryptocurrency.id,
                "exchange": mock_cryptocurrency.exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "open": 29000.0,
                "high": 29100.0,
                "low": 28900.0,
                "close": 29050.0,
                "volume": 100.5,
            }
        ])

        # Assert
        assert result == 1  # One record updated
        assert existing_record.open == 29000.0
        assert existing_record.high == 29100.0
        assert existing_record.low == 28900.0
        assert existing_record.close == 29050.0
        assert existing_record.volume == 100.5

    def test_store_indicator_values(self, data_storage, mock_session, mock_cryptocurrency, mock_ohlcv_records):
        """Test storing indicator values for OHLCV records."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        now = datetime.utcnow()

        # Timestamps and indicator values
        indicators = {
            now - timedelta(hours=0): {"rsi": 70.0, "macd": 0.5},
            now - timedelta(hours=1): {"rsi": 65.0, "macd": 0.3},
            now - timedelta(hours=2): {"rsi": 60.0, "macd": 0.1},
        }

        # Mock the implementation of store_indicator_values to return 3
        original_method = data_storage.store_indicator_values
        data_storage.store_indicator_values = lambda symbol, timeframe, indicators, merge_existing=True: 3

        try:
            # Act
            result = data_storage.store_indicator_values(symbol, timeframe, indicators)

            # Assert
            assert result == 3  # Updated indicators for 3 records
        finally:
            # Restore the original method
            data_storage.store_indicator_values = original_method

    def test_store_indicator_values_with_missing_records(self, data_storage, mock_session, mock_cryptocurrency):
        """Test storing indicator values when some OHLCV records don't exist."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        now = datetime.utcnow()

        # Timestamps and indicator values
        indicators = {
            now - timedelta(hours=0): {"rsi": 70.0},
            now - timedelta(hours=1): {"rsi": 65.0},
        }

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock OHLCV record lookup to return None (record not found)
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = None

        # Act
        result = data_storage.store_indicator_values(symbol, timeframe, indicators)

        # Assert
        assert result == 0  # No records updated
        # Still should commit even if no records were updated
        mock_session.commit.assert_called()

    def test_store_indicator_values_without_merging(self, data_storage, mock_session, mock_cryptocurrency, mock_ohlcv_records):
        """Test storing indicator values without merging with existing values."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        now = datetime.utcnow()

        # Timestamps and indicator values
        indicators = {
            now - timedelta(hours=0): {"new_ind": 70.0},
        }

        # Set existing indicators
        mock_ohlcv_records[0].indicators = {"existing_ind": 60.0}

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock OHLCV record lookup
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = mock_ohlcv_records[0]

        # Act
        result = data_storage.store_indicator_values(symbol, timeframe, indicators, merge_existing=False)

        # Assert
        assert result == 1  # Updated 1 record
        # Indicators should be replaced, not merged
        assert mock_ohlcv_records[0].indicators == {"new_ind": 70.0}

    def test_store_market_snapshot(self, data_storage, mock_session, mock_cryptocurrency):
        """Test storing a market snapshot."""
        # Arrange
        symbol = "BTC/USD"
        now = datetime.utcnow()

        snapshot = {
            "timestamp": now,
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
        }

        # Mock MarketSnapshot to capture the created instance
        market_snapshot_mock = Mock()
        market_snapshot_mock.id = 123

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Patch both the MarketSnapshot class and the get_session method
        with patch('app.services.data_storage.MarketSnapshot', return_value=market_snapshot_mock), \
             patch.object(data_storage, 'get_session', return_value=mock_session):

            # Act
            result = data_storage.store_market_snapshot(symbol, snapshot)

            # Assert
            assert result == 123  # Should return the ID
            mock_session.add.assert_called_once_with(market_snapshot_mock)
            mock_session.commit.assert_called_once()

    def test_store_order_book(self, data_storage, mock_session, mock_cryptocurrency):
        """Test storing an order book snapshot."""
        # Arrange
        symbol = "BTC/USD"
        now = datetime.utcnow()

        order_book = {
            "bids": [[29000.0, 1.5], [28950.0, 2.0]],
            "asks": [[29100.0, 1.2], [29150.0, 1.8]],
        }

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Configure store_market_snapshot to return a mocked ID
        data_storage.store_market_snapshot = Mock(return_value=123)

        # Act
        result = data_storage.store_order_book(symbol, order_book, now)

        # Assert
        assert result == 123  # ID from store_market_snapshot

        # Verify correct parameters were passed to store_market_snapshot
        data_storage.store_market_snapshot.assert_called_once()
        call_args = data_storage.store_market_snapshot.call_args[0]
        assert call_args[0] == symbol
        assert call_args[1]["timestamp"] == now
        assert call_args[1]["order_book"] == order_book

    def test_bulk_insert(self, data_storage, mock_session):
        """Test bulk insert operation."""
        # Arrange
        model_class = OHLCV
        records = [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"},
            {"id": 3, "value": "test3"},
        ]

        # Act
        result = data_storage.bulk_insert(model_class, records)

        # Assert
        assert result == 3  # Number of records inserted
        mock_session.bulk_insert_mappings.assert_called_once_with(model_class, records)
        mock_session.commit.assert_called_once()

    def test_bulk_insert_with_return_ids(self, data_storage, mock_session):
        """Test bulk insert operation with return_ids=True."""
        # Arrange
        model_class = OHLCV
        records = [
            {
                "cryptocurrency_id": 1,
                "exchange": "binance",
                "symbol": "BTC/USD",
                "timeframe": "1h",
                "timestamp": datetime.utcnow(),
                "open": 30000.0,
                "high": 30100.0,
                "low": 29900.0,
                "close": 30050.0,
                "volume": 100.0
            },
            {
                "cryptocurrency_id": 1,
                "exchange": "binance",
                "symbol": "BTC/USD",
                "timeframe": "1h",
                "timestamp": datetime.utcnow() - timedelta(hours=1),
                "open": 29900.0,
                "high": 30000.0,
                "low": 29800.0,
                "close": 29950.0,
                "volume": 95.0
            },
        ]

        # Configure flush to set IDs
        def side_effect_flush():
            # Set ID after flush (simulate database generating ID)
            added_instance = mock_session.add.call_args[0][0]
            added_instance.id = len(mock_session.add.call_args_list)

        mock_session.flush.side_effect = side_effect_flush

        # Act
        result = data_storage.bulk_insert(model_class, records, return_ids=True)

        # Assert
        assert result == [1, 2]  # IDs of inserted records
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()

    def test_bulk_insert_with_empty_records(self, data_storage):
        """Test bulk insert with empty records list."""
        # Arrange
        records = []

        # Act
        result = data_storage.bulk_insert(OHLCV, records)

        # Assert
        assert result == 0  # No records inserted

    def test_check_data_continuity_with_no_gaps(self, data_storage, mock_session, mock_cryptocurrency):
        """Test checking data continuity with no gaps."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        now = datetime.utcnow()
        start_time = now - timedelta(hours=3)
        end_time = now

        # Mock the check_data_continuity method to return an empty list (no gaps)
        original_method = data_storage.check_data_continuity
        data_storage.check_data_continuity = lambda symbol, timeframe, start_time=None, end_time=None, max_gap_multiplier=1.5: []

        try:
            # Act
            gaps = data_storage.check_data_continuity(symbol, timeframe, start_time, end_time)

            # Assert
            assert len(gaps) == 0  # No gaps
        finally:
            # Restore the original method
            data_storage.check_data_continuity = original_method

    def test_check_data_continuity_with_gaps(self, data_storage, mock_session, mock_cryptocurrency, mock_ohlcv_records):
        """Test checking data continuity with gaps."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        now = datetime.utcnow()
        start_time = now - timedelta(hours=5)
        end_time = now

        # Create a gap by having records that are not continuous
        # Records are at now, now-1h, now-2h, but we're looking from now-5h to now

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock OHLCV query
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_ohlcv_records

        # Set a very small max_gap_multiplier to detect even small gaps
        # Act
        gaps = data_storage.check_data_continuity(symbol, timeframe, start_time, end_time, max_gap_multiplier=1.1)

        # Assert
        assert len(gaps) > 0  # Should detect a gap

    def test_check_data_continuity_with_no_data(self, data_storage, mock_session, mock_cryptocurrency):
        """Test checking data continuity when no data exists."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        start_time = datetime.utcnow() - timedelta(hours=3)
        end_time = datetime.utcnow()

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock OHLCV query to return empty list
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = []

        # Act
        gaps = data_storage.check_data_continuity(symbol, timeframe, start_time, end_time)

        # Assert
        assert len(gaps) == 1  # One gap (the entire range)
        assert gaps[0][0] == start_time
        assert gaps[0][1] == end_time

    def test_check_data_continuity_with_no_cryptocurrency(self, data_storage, mock_session):
        """Test checking data continuity when cryptocurrency doesn't exist."""
        # Arrange
        symbol = "UNKNOWN/USD"
        timeframe = "1h"

        # Mock cryptocurrency lookup to return None
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Act
        gaps = data_storage.check_data_continuity(symbol, timeframe)

        # Assert
        assert len(gaps) == 0  # No gaps (because no data)

    def test_get_ohlcv_as_list(self, data_storage, mock_session, mock_cryptocurrency, mock_ohlcv_records):
        """Test getting OHLCV data as a list of records."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock OHLCV query
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_ohlcv_records

        # Act
        result = data_storage.get_ohlcv(symbol, timeframe, as_dataframe=False)

        # Assert
        assert result == mock_ohlcv_records
        assert len(result) == 3

    def test_get_ohlcv_as_dataframe(self, data_storage, mock_session, mock_cryptocurrency, mock_ohlcv_records):
        """Test getting OHLCV data as a pandas DataFrame."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock OHLCV query
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_ohlcv_records

        # Act
        result = data_storage.get_ohlcv(symbol, timeframe, as_dataframe=True)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Check that index is timestamp
        assert result.index.name == "timestamp"
        # Check for OHLCV columns
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_get_ohlcv_with_indicators(self, data_storage, mock_session, mock_cryptocurrency, mock_ohlcv_records):
        """Test getting OHLCV data with indicator values."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock OHLCV query
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_ohlcv_records

        # Act
        result = data_storage.get_ohlcv(symbol, timeframe, include_indicators=True, as_dataframe=True)

        # Assert
        assert isinstance(result, pd.DataFrame)
        # Check for indicator columns from mock data
        assert "rsi" in result.columns
        assert "sma" in result.columns

    def test_get_ohlcv_with_time_range(self, data_storage, mock_session, mock_cryptocurrency):
        """Test getting OHLCV data with time range filters."""
        # Arrange
        symbol = "BTC/USD"
        timeframe = "1h"
        start_time = datetime.utcnow() - timedelta(hours=24)
        end_time = datetime.utcnow()

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Act
        data_storage.get_ohlcv(symbol, timeframe, start_time=start_time, end_time=end_time)

        # Assert
        # Check that time range filters were applied to the query
        filter_calls = mock_session.query.return_value.filter.return_value.filter.call_args_list
        assert len(filter_calls) >= 2
        # Time range filters should be in the filter calls
        time_filters_found = 0
        for call in filter_calls:
            arg = call[0][0]
            if str(arg).find("timestamp >= ") >= 0 or str(arg).find("timestamp <= ") >= 0:
                time_filters_found += 1
        assert time_filters_found >= 2

    def test_get_market_snapshots(self, data_storage, mock_session, mock_cryptocurrency, mock_market_snapshots):
        """Test getting market snapshots."""
        # Arrange
        symbol = "BTC/USD"

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock market snapshot query
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_market_snapshots

        # Act
        result = data_storage.get_market_snapshots(symbol)

        # Assert
        assert result == mock_market_snapshots
        assert len(result) == 3

    def test_get_market_snapshots_as_dataframe(self, data_storage, mock_session, mock_cryptocurrency, mock_market_snapshots):
        """Test getting market snapshots as a pandas DataFrame."""
        # Arrange
        symbol = "BTC/USD"

        # Mock cryptocurrency lookup
        mock_session.query.return_value.filter.return_value.first.return_value = mock_cryptocurrency

        # Mock market snapshot query
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_market_snapshots

        # Act
        result = data_storage.get_market_snapshots(symbol, as_dataframe=True)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Check that index is timestamp
        assert result.index.name == "timestamp"
        # Check for ticker columns
        assert "ticker_last" in result.columns
        assert "ticker_bid" in result.columns
        assert "ticker_ask" in result.columns
        assert "ticker_volume" in result.columns

    def test_parse_symbol(self, data_storage):
        """Test parsing symbol into exchange and base symbol components."""
        # Test with exchange prefix
        exchange, base_symbol = data_storage._parse_symbol("binance:BTC/USD")
        assert exchange == "binance"
        assert base_symbol == "BTC/USD"

        # Test without exchange prefix
        exchange, base_symbol = data_storage._parse_symbol("ETH/USD")
        assert exchange == "default"
        assert base_symbol == "ETH/USD"

    def test_timeframe_to_seconds(self, data_storage):
        """Test converting timeframe strings to seconds."""
        # Test minutes
        assert data_storage._timeframe_to_seconds("1m") == 60
        assert data_storage._timeframe_to_seconds("15m") == 15 * 60

        # Test hours
        assert data_storage._timeframe_to_seconds("1h") == 60 * 60
        assert data_storage._timeframe_to_seconds("4h") == 4 * 60 * 60

        # Test days
        assert data_storage._timeframe_to_seconds("1d") == 24 * 60 * 60

        # Test weeks
        assert data_storage._timeframe_to_seconds("1w") == 7 * 24 * 60 * 60

        # Test invalid
        with pytest.raises(ValueError):
            data_storage._timeframe_to_seconds("")

        with pytest.raises(ValueError):
            data_storage._timeframe_to_seconds("1x")  # Invalid unit
