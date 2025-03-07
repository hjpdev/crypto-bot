"""Tests for the MarketDataCollector class."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from app.core.exceptions import APIError
from app.core.scheduler import TaskScheduler
from app.tasks.market_data_collector import MarketDataCollector


class TestMarketDataCollector:
    """Tests for the MarketDataCollector class."""

    @pytest.fixture
    def mock_exchange_service(self):
        """Fixture to provide a mock exchange service."""
        mock = MagicMock()
        mock.is_valid_symbol.return_value = True
        mock.fetch_ohlcv.return_value = [
            # timestamp, open, high, low, close, volume
            [1614556800000, 100.0, 105.0, 95.0, 102.0, 1000.0],
            [1614556900000, 102.0, 107.0, 101.0, 106.0, 1500.0],
        ]
        return mock

    @pytest.fixture
    def mock_storage_service(self):
        """Fixture to provide a mock storage service."""
        return MagicMock()

    @pytest.fixture
    def mock_config_service(self):
        """Fixture to provide a mock config service."""
        return MagicMock()

    @pytest.fixture
    def mock_logger(self):
        """Fixture to provide a mock logger."""
        return MagicMock()

    @pytest.fixture
    def collector(self, mock_exchange_service, mock_storage_service,
                  mock_config_service, mock_logger):
        """Fixture to create a MarketDataCollector with mocked dependencies."""
        return MarketDataCollector(
            exchange_service=mock_exchange_service,
            storage_service=mock_storage_service,
            config_service=mock_config_service,
            interval_minutes=5,
            max_retries=3,
            backfill_missing=True,
            logger=mock_logger,
        )

    def test_init(self, collector, mock_exchange_service, mock_storage_service,
                  mock_config_service, mock_logger):
        """Test initializing the MarketDataCollector."""
        assert collector._exchange_service == mock_exchange_service
        assert collector._storage_service == mock_storage_service
        assert collector._config_service == mock_config_service
        assert collector._interval_minutes == 5
        assert collector._max_retries == 3
        assert collector._backfill_missing is True
        assert collector._logger == mock_logger
        assert collector._active_symbols == set()
        assert collector._last_collection_time == {}
        assert collector._collection_stats == {}
        assert collector._error_counts == {}

    def test_register_with_scheduler(self, collector, mock_logger):
        """Test registering tasks with the scheduler."""
        scheduler = TaskScheduler(logger=mock_logger)

        task_name = collector.register_with_scheduler(scheduler)

        assert task_name == "market_data_collection"
        assert "market_data_collection" in scheduler._tasks
        assert "cryptocurrency_metadata_update" in scheduler._tasks
        assert scheduler._tasks["market_data_collection"].interval == 300  # 5 minutes in seconds
        assert scheduler._tasks["cryptocurrency_metadata_update"].interval == 86400  # 24 hours in seconds
        assert scheduler._tasks["market_data_collection"].priority == 10
        assert scheduler._tasks["cryptocurrency_metadata_update"].priority == 50

    def test_add_symbols(self, collector, mock_exchange_service):
        """Test adding symbols to the collector."""
        # Valid symbols
        collector.add_symbols(["BTC/USDT", "ETH/USDT"])

        assert "BTC/USDT" in collector._active_symbols
        assert "ETH/USDT" in collector._active_symbols
        assert "BTC/USDT" in collector._collection_stats
        assert "ETH/USDT" in collector._collection_stats

        # Check collection stats initialization
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            stats = collector._collection_stats[symbol]
            assert stats["total_collections"] == 0
            assert stats["successful_collections"] == 0
            assert stats["failed_collections"] == 0
            assert stats["last_successful_time"] is None
            assert stats["last_error"] is None
            assert stats["last_error_time"] is None

        # Test with invalid symbol
        mock_exchange_service.is_valid_symbol.side_effect = lambda s: s != "INVALID/PAIR"
        collector.add_symbols(["INVALID/PAIR", "LTC/USDT"])

        assert "INVALID/PAIR" not in collector._active_symbols
        assert "LTC/USDT" in collector._active_symbols

    def test_remove_symbols(self, collector):
        """Test removing symbols from the collector."""
        collector.add_symbols(["BTC/USDT", "ETH/USDT", "LTC/USDT"])

        collector.remove_symbols(["ETH/USDT", "XRP/USDT"])  # XRP/USDT not in list

        assert "BTC/USDT" in collector._active_symbols
        assert "ETH/USDT" not in collector._active_symbols
        assert "LTC/USDT" in collector._active_symbols
        assert len(collector._active_symbols) == 2

    def test_get_active_symbols(self, collector):
        """Test getting the list of active symbols."""
        collector.add_symbols(["BTC/USDT", "ETH/USDT"])

        symbols = collector.get_active_symbols()

        assert len(symbols) == 2
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols

    def test_get_collection_stats(self, collector):
        """Test getting collection statistics."""
        collector.add_symbols(["BTC/USDT", "ETH/USDT"])

        # Update some statistics manually
        collector._collection_stats["BTC/USDT"]["total_collections"] = 5
        collector._collection_stats["BTC/USDT"]["successful_collections"] = 5
        collector._collection_stats["ETH/USDT"]["total_collections"] = 3
        collector._collection_stats["ETH/USDT"]["failed_collections"] = 1

        stats = collector.get_collection_stats()

        assert stats["BTC/USDT"]["total_collections"] == 5
        assert stats["BTC/USDT"]["successful_collections"] == 5
        assert stats["ETH/USDT"]["total_collections"] == 3
        assert stats["ETH/USDT"]["failed_collections"] == 1

    def test_run_no_symbols(self, collector, mock_logger):
        """Test running collection with no symbols configured."""
        result = collector.run()

        assert result is True
        mock_logger.info.assert_any_call("No active symbols configured for collection")

    def test_run_with_symbols(self, collector, mock_exchange_service,
                              mock_storage_service, monkeypatch):
        """Test running collection with configured symbols."""
        collector.add_symbols(["BTC/USDT", "ETH/USDT"])

        # Mock the collect_data method to track calls
        mock_collect_data = MagicMock(return_value=True)
        monkeypatch.setattr(collector, "collect_data", mock_collect_data)

        result = collector.run()

        assert result is True
        # Verify that collect_data was called with the symbols (order may vary since active_symbols is a set)
        mock_collect_data.assert_called_once()
        call_args = mock_collect_data.call_args[0][0]
        assert len(call_args) == 2
        assert set(call_args) == {"BTC/USDT", "ETH/USDT"}

    def test_run_with_many_symbols(self, collector, monkeypatch):
        """Test running collection with many symbols (should batch them)."""
        # Add more than batch_size symbols
        symbols = [f"SYMBOL{i}/USDT" for i in range(15)]
        collector.add_symbols(symbols)

        # Mock the collect_data method to track calls
        mock_collect_data = MagicMock(return_value=True)
        monkeypatch.setattr(collector, "collect_data", mock_collect_data)

        result = collector.run()

        assert result is True
        # Should have been called with two batches
        assert mock_collect_data.call_count == 2
        # First batch should have 10 symbols
        assert len(mock_collect_data.call_args_list[0][0][0]) == 10
        # Second batch should have 5 symbols
        assert len(mock_collect_data.call_args_list[1][0][0]) == 5

    def test_collect_data_success(self, collector, mock_exchange_service, mock_storage_service):
        """Test successful data collection."""
        collector.add_symbols(["BTC/USDT"])

        result = collector.collect_data(["BTC/USDT"])

        assert result is True
        mock_exchange_service.fetch_ohlcv.assert_called_once()
        mock_storage_service.store_ohlcv_data.assert_called_once()

        # Check statistics updates
        stats = collector._collection_stats["BTC/USDT"]
        assert stats["total_collections"] == 1
        assert stats["successful_collections"] == 1
        assert stats["failed_collections"] == 0
        assert stats["last_successful_time"] is not None

        # Check that last collection time was updated
        assert "BTC/USDT" in collector._last_collection_time

    def test_collect_data_error(self, collector, mock_exchange_service, mock_logger):
        """Test handling of collection errors."""
        collector.add_symbols(["BTC/USDT"])

        # Simulate API error
        mock_exchange_service.fetch_ohlcv.side_effect = APIError("API connection error")

        result = collector.collect_data(["BTC/USDT"])

        assert result is False
        mock_logger.error.assert_called()

        # Check statistics updates
        stats = collector._collection_stats["BTC/USDT"]
        assert stats["total_collections"] == 1
        assert stats["successful_collections"] == 0
        assert stats["failed_collections"] == 1
        assert stats["last_error"] == "API connection error"
        assert stats["last_error_time"] is not None

        # Check error count tracking
        assert collector._error_counts["BTC/USDT"] == 1

    def test_collect_data_backfill(self, collector, mock_exchange_service, monkeypatch):
        """Test backfilling of missing data."""
        collector.add_symbols(["BTC/USDT"])

        # Set a last collection time that was a while ago
        one_hour_ago = datetime.now() - timedelta(hours=1)
        collector._last_collection_time["BTC/USDT"] = one_hour_ago

        # Mock backfill method to track calls
        mock_backfill = MagicMock()
        monkeypatch.setattr(collector, "_backfill_data", mock_backfill)

        result = collector.collect_data(["BTC/USDT"])

        assert result is True
        # Should have called backfill with appropriate parameters
        mock_backfill.assert_called_once()
        # First arg should be symbol
        assert mock_backfill.call_args[0][0] == "BTC/USDT"
        # Second arg should be the last collection time
        assert mock_backfill.call_args[0][1] == one_hour_ago

    def test_backfill_data(self, collector, mock_exchange_service, mock_storage_service):
        """Test the backfill data functionality."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        collector._backfill_data("BTC/USDT", start_time, end_time)

        # Should call fetch_ohlcv with appropriate parameters
        mock_exchange_service.fetch_ohlcv.assert_called_once()
        call_args = mock_exchange_service.fetch_ohlcv.call_args
        # The first positional argument is the symbol
        assert call_args[0][0] == "BTC/USDT"
        # Check the keyword arguments
        assert "since" in call_args[1]
        assert call_args[1]["since"] == int(start_time.timestamp() * 1000)
        assert call_args[1]["limit"] == 100

        # Should process and store the data
        mock_storage_service.store_ohlcv_data.assert_called_once()

    def test_update_cryptocurrency_metadata(self, collector, mock_exchange_service, mock_storage_service):
        """Test updating cryptocurrency metadata."""
        collector.add_symbols(["BTC/USDT", "ETH/USDT"])

        # Mock the fetch_currency_info method
        mock_exchange_service.fetch_currency_info.return_value = {"market_cap": 1000000}

        result = collector.update_cryptocurrency_metadata()

        assert result is True
        # Should call fetch_currency_info for each unique currency
        assert mock_exchange_service.fetch_currency_info.call_count == 3  # BTC, ETH, USDT
        # Should call update_cryptocurrency_metadata for each currency
        assert mock_storage_service.update_cryptocurrency_metadata.call_count == 3

    def test_process_ohlcv_data(self, collector, mock_storage_service):
        """Test processing of OHLCV data."""
        # Sample OHLCV data
        ohlcv_data = [
            [1614556800000, 100.0, 105.0, 95.0, 102.0, 1000.0],
            [1614556900000, 102.0, 107.0, 101.0, 106.0, 1500.0],
        ]

        collector._process_ohlcv_data("BTC/USDT", ohlcv_data)

        # Check that data was processed and stored
        mock_storage_service.store_ohlcv_data.assert_called_once()
        stored_data = mock_storage_service.store_ohlcv_data.call_args[0][0]

        # Verify the processed data
        assert len(stored_data) == 2
        assert stored_data[0]["symbol"] == "BTC/USDT"
        assert stored_data[0]["open"] == 100.0
        assert stored_data[0]["high"] == 105.0
        assert stored_data[0]["low"] == 95.0
        assert stored_data[0]["close"] == 102.0
        assert stored_data[0]["volume"] == 1000.0
        assert isinstance(stored_data[0]["timestamp"], datetime)

        assert stored_data[1]["symbol"] == "BTC/USDT"
        assert stored_data[1]["open"] == 102.0
        assert stored_data[1]["volume"] == 1500.0

    def test_get_exchange_timeframe(self, collector):
        """Test conversion of interval to exchange timeframe format."""
        # Test with minutes
        collector._interval_minutes = 5
        assert collector._get_exchange_timeframe() == "5m"

        # Test with hours
        collector._interval_minutes = 60
        assert collector._get_exchange_timeframe() == "1h"

        # Test with days
        collector._interval_minutes = 1440
        assert collector._get_exchange_timeframe() == "1d"

        # Test with seconds
        collector._interval_minutes = 0.5  # 30 seconds
        assert collector._get_exchange_timeframe() == "30s"

    def test_error_recovery(self, collector, mock_exchange_service, monkeypatch):
        """Test error recovery after failures."""
        collector.add_symbols(["BTC/USDT"])

        # First call fails
        mock_exchange_service.fetch_ohlcv.side_effect = [
            APIError("Temporary error"),  # First call fails
            [[1614556800000, 100.0, 105.0, 95.0, 102.0, 1000.0]],  # Second call succeeds
        ]

        # First collection (fails)
        result = collector.collect_data(["BTC/USDT"])
        assert result is False
        assert collector._error_counts["BTC/USDT"] == 1

        # Reset side effect to ensure next call succeeds
        mock_exchange_service.fetch_ohlcv.side_effect = None
        mock_exchange_service.fetch_ohlcv.return_value = [[1614556800000, 100.0, 105.0, 95.0, 102.0, 1000.0]]

        # Second collection (succeeds)
        result = collector.collect_data(["BTC/USDT"])
        assert result is True

        # Error count should be reset
        assert collector._error_counts["BTC/USDT"] == 0
