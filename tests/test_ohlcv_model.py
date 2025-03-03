import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from sqlalchemy.exc import IntegrityError

from app.models.ohlcv import OHLCV
from app.models.cryptocurrency import Cryptocurrency
from app.core.exceptions import ValidationError


class TestOHLCVModel:
    """Tests for the OHLCV model."""

    def test_create_ohlcv(self, db_session, sample_crypto):
        """Test creating a new OHLCV record."""
        timestamp = datetime.now(timezone.utc)
        ohlcv = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="coinbase",
            symbol=sample_crypto.symbol,
            timeframe="1h",
            timestamp=timestamp,
            open=Decimal("50000.00"),
            high=Decimal("52000.00"),
            low=Decimal("49500.00"),
            close=Decimal("51500.00"),
            volume=Decimal("1000.50"),
        )
        db_session.add(ohlcv)
        db_session.commit()

        saved_ohlcv = db_session.query(OHLCV).filter_by(id=ohlcv.id).first()
        assert saved_ohlcv is not None
        assert saved_ohlcv.cryptocurrency_id == sample_crypto.id
        assert saved_ohlcv.exchange == "coinbase"
        assert saved_ohlcv.symbol == sample_crypto.symbol
        if saved_ohlcv.timestamp.tzinfo is None:
            saved_timestamp = saved_ohlcv.timestamp.replace(tzinfo=timezone.utc)
        else:
            saved_timestamp = saved_ohlcv.timestamp
        assert saved_timestamp == timestamp
        assert saved_ohlcv.open == Decimal("50000.00")
        assert saved_ohlcv.high == Decimal("52000.00")
        assert saved_ohlcv.low == Decimal("49500.00")
        assert saved_ohlcv.close == Decimal("51500.00")
        assert saved_ohlcv.volume == Decimal("1000.50")
        assert saved_ohlcv.indicators is None

    def test_update_ohlcv(self, db_session, sample_ohlcv):
        """Test updating an OHLCV record."""
        sample_ohlcv.close = Decimal("52000.00")
        sample_ohlcv.volume = Decimal("1200.75")
        db_session.add(sample_ohlcv)
        db_session.commit()

        updated_ohlcv = db_session.query(OHLCV).filter_by(id=sample_ohlcv.id).first()
        assert updated_ohlcv.close == Decimal("52000.00")
        assert updated_ohlcv.volume == Decimal("1200.75")

    def test_delete_ohlcv(self, db_session, sample_ohlcv):
        """Test deleting an OHLCV record."""
        db_session.delete(sample_ohlcv)
        db_session.commit()

        deleted_ohlcv = db_session.query(OHLCV).filter_by(id=sample_ohlcv.id).first()
        assert deleted_ohlcv is None

    def test_relationship_with_cryptocurrency(self, db_session, sample_crypto, sample_ohlcv):
        """Test relationship between OHLCV and Cryptocurrency."""
        # Test accessing cryptocurrency from OHLCV
        ohlcv_from_db = db_session.query(OHLCV).filter_by(id=sample_ohlcv.id).first()
        assert ohlcv_from_db.cryptocurrency is not None
        assert ohlcv_from_db.cryptocurrency.id == sample_crypto.id
        assert ohlcv_from_db.cryptocurrency.symbol.startswith("BTC/USD")

        # Test accessing OHLCV data from cryptocurrency
        crypto_from_db = db_session.query(Cryptocurrency).filter_by(id=sample_crypto.id).first()
        assert crypto_from_db.market_data is not None
        assert len(crypto_from_db.market_data) == 1
        assert crypto_from_db.market_data[0].id == sample_ohlcv.id

    def test_cascade_delete(self, db_session, sample_crypto, sample_ohlcv):
        """Test that deleting a cryptocurrency cascades to OHLCV records."""
        db_session.delete(sample_crypto)
        db_session.commit()

        # Verify OHLCV record was also deleted
        deleted_ohlcv = db_session.query(OHLCV).filter_by(id=sample_ohlcv.id).first()
        assert deleted_ohlcv is None

    def test_unique_constraint(self, db_session, sample_crypto, sample_ohlcv):
        """Test that the unique constraint is enforced."""
        # Create another OHLCV with the same exchange, symbol, timeframe and timestamp
        duplicate_ohlcv = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange=sample_ohlcv.exchange,
            symbol=sample_ohlcv.symbol,
            timeframe=sample_ohlcv.timeframe,
            timestamp=sample_ohlcv.timestamp,
            open=Decimal("50000.00"),
            high=Decimal("52000.00"),
            low=Decimal("49500.00"),
            close=Decimal("51500.00"),
            volume=Decimal("1000.50"),
        )

        db_session.add(duplicate_ohlcv)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_indicators_json_storage(self, db_session, sample_ohlcv):
        """Test storing and retrieving indicator data in JSON format."""
        # Test initial JSON data
        assert sample_ohlcv.indicators is not None
        assert "rsi" in sample_ohlcv.indicators
        assert sample_ohlcv.indicators["rsi"] == 65.5
        assert "macd" in sample_ohlcv.indicators
        assert sample_ohlcv.indicators["macd"]["signal"] == 0.5

        # Update indicators
        sample_ohlcv.update_indicators(db_session, {
            "bollinger_bands": {
                "upper": 55000,
                "middle": 52000,
                "lower": 49000
            },
            "rsi": 70  # Should update existing value
        })

        # Verify indicators were updated
        updated_ohlcv = db_session.query(OHLCV).filter_by(id=sample_ohlcv.id).first()
        assert updated_ohlcv.indicators["rsi"] == 70  # Updated value
        assert updated_ohlcv.indicators["macd"]["signal"] == 0.5  # Preserved value
        assert "bollinger_bands" in updated_ohlcv.indicators  # New indicator
        assert updated_ohlcv.indicators["bollinger_bands"]["upper"] == 55000

    def test_data_validation(self):
        """Test the validate_ohlcv_data static method."""
        # Valid data
        OHLCV.validate_ohlcv_data(
            open_price=50000,
            high_price=52000,
            low_price=49000,
            close_price=51000,
            volume=1000
        )

        # Test invalid cases
        with pytest.raises(ValidationError, match="numbers"):
            OHLCV.validate_ohlcv_data("50000", 52000, 49000, 51000, 1000)

        with pytest.raises(ValidationError, match="High price cannot be lower than low price"):
            OHLCV.validate_ohlcv_data(50000, 49000, 52000, 51000, 1000)

        with pytest.raises(ValidationError, match="negative"):
            OHLCV.validate_ohlcv_data(50000, 52000, -1, 51000, 1000)

        with pytest.raises(ValidationError, match="negative"):
            OHLCV.validate_ohlcv_data(50000, 52000, 49000, 51000, -5)

        with pytest.raises(ValidationError, match="inconsistent"):
            # High < open, which is inconsistent
            OHLCV.validate_ohlcv_data(53000, 52000, 49000, 51000, 1000)

        with pytest.raises(ValidationError, match="inconsistent"):
            # Low > close, which is inconsistent
            OHLCV.validate_ohlcv_data(50000, 52000, 52500, 51000, 1000)

    def test_get_latest(self, db_session, sample_crypto):
        """Test the get_latest class method."""
        # Create multiple OHLCV records with different timestamps
        now = datetime.now(timezone.utc)

        ohlcv1 = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="binance",
            symbol=sample_crypto.symbol,
            timeframe="1h",
            timestamp=now - timedelta(hours=3),
            open=Decimal("49000.00"),
            high=Decimal("50000.00"),
            low=Decimal("48500.00"),
            close=Decimal("49500.00"),
            volume=Decimal("1000.50"),
        )

        ohlcv2 = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="binance",
            symbol=sample_crypto.symbol,
            timeframe="1h",
            timestamp=now - timedelta(hours=2),
            open=Decimal("49500.00"),
            high=Decimal("51000.00"),
            low=Decimal("49000.00"),
            close=Decimal("50500.00"),
            volume=Decimal("1100.50"),
        )

        ohlcv3 = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="binance",
            symbol=sample_crypto.symbol,
            timeframe="1h",
            timestamp=now - timedelta(hours=1),
            open=Decimal("50500.00"),
            high=Decimal("52000.00"),
            low=Decimal("50000.00"),
            close=Decimal("51500.00"),
            volume=Decimal("1200.50"),
        )

        # Different timeframe
        ohlcv4 = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="binance",
            symbol=sample_crypto.symbol,
            timeframe="5m",
            timestamp=now - timedelta(minutes=5),
            open=Decimal("51500.00"),
            high=Decimal("51800.00"),
            low=Decimal("51400.00"),
            close=Decimal("51700.00"),
            volume=Decimal("500.25"),
        )

        db_session.add_all([ohlcv1, ohlcv2, ohlcv3, ohlcv4])
        db_session.commit()

        # Test get_latest with default parameters
        latest_data = OHLCV.get_latest(db_session, sample_crypto.symbol)

        assert len(latest_data) == 3  # Should get all 1h records
        # Should be in reverse chronological order
        assert latest_data[0].timestamp > latest_data[1].timestamp
        assert latest_data[1].timestamp > latest_data[2].timestamp

        # Test get_latest with limit
        limited_data = OHLCV.get_latest(db_session, sample_crypto.symbol, limit=2)
        assert len(limited_data) == 2

        # Test get_latest with different timeframe
        timeframe_data = OHLCV.get_latest(db_session, sample_crypto.symbol, timeframe="5m")
        assert len(timeframe_data) == 1
        assert timeframe_data[0].timeframe == "5m"

    def test_get_range(self, db_session, sample_crypto):
        """Test the get_range class method."""
        # Create multiple OHLCV records with different timestamps
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=3)
        middle_time = now - timedelta(hours=2)
        end_time = now - timedelta(hours=1)

        ohlcv1 = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="binance",
            symbol=sample_crypto.symbol,
            timeframe="1h",
            timestamp=start_time,
            open=Decimal("49000.00"),
            high=Decimal("50000.00"),
            low=Decimal("48500.00"),
            close=Decimal("49500.00"),
            volume=Decimal("1000.50"),
        )

        ohlcv2 = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="binance",
            symbol=sample_crypto.symbol,
            timeframe="1h",
            timestamp=middle_time,
            open=Decimal("49500.00"),
            high=Decimal("51000.00"),
            low=Decimal("49000.00"),
            close=Decimal("50500.00"),
            volume=Decimal("1100.50"),
        )

        ohlcv3 = OHLCV(
            cryptocurrency_id=sample_crypto.id,
            exchange="binance",
            symbol=sample_crypto.symbol,
            timeframe="1h",
            timestamp=end_time,
            open=Decimal("50500.00"),
            high=Decimal("52000.00"),
            low=Decimal("50000.00"),
            close=Decimal("51500.00"),
            volume=Decimal("1200.50"),
        )

        db_session.add_all([ohlcv1, ohlcv2, ohlcv3])
        db_session.commit()

        # Test get_range for full range
        full_range = OHLCV.get_range(
            db_session,
            sample_crypto.symbol,
            start=start_time - timedelta(minutes=5),  # Ensure we capture the first record
            end=end_time + timedelta(minutes=5)  # Ensure we capture the last record
        )

        assert len(full_range) == 3
        # Compare timestamps accounting for potential timezone differences
        if full_range[0].timestamp.tzinfo is None:
            saved_timestamp = full_range[0].timestamp.replace(tzinfo=timezone.utc)
        else:
            saved_timestamp = full_range[0].timestamp
        assert saved_timestamp == start_time

        if full_range[2].timestamp.tzinfo is None:
            saved_timestamp = full_range[2].timestamp.replace(tzinfo=timezone.utc)
        else:
            saved_timestamp = full_range[2].timestamp
        assert saved_timestamp == end_time

        OHLCV.get_range(
            db_session,
            sample_crypto.symbol,
            start=middle_time,
            end=end_time
        )

        # Test validation
        with pytest.raises(ValidationError, match="End time must be after start time"):
            OHLCV.get_range(
                db_session,
                sample_crypto.id,
                start=end_time,
                end=start_time
            )

    def test_as_dict_method(self, db_session, sample_ohlcv):
        """Test the as_dict property."""
        data_dict = sample_ohlcv.as_dict

        assert isinstance(data_dict, dict)
        assert data_dict["id"] == sample_ohlcv.id
        assert data_dict["cryptocurrency_id"] == sample_ohlcv.cryptocurrency_id
        assert data_dict["exchange"] == sample_ohlcv.exchange
        assert data_dict["symbol"] == sample_ohlcv.symbol
        assert data_dict["timeframe"] == sample_ohlcv.timeframe
        assert isinstance(data_dict["timestamp"], str)  # ISO format string
        assert isinstance(data_dict["open"], float)
        assert isinstance(data_dict["high"], float)
        assert isinstance(data_dict["close"], float)
        assert isinstance(data_dict["low"], float)
        assert isinstance(data_dict["volume"], float)
        assert isinstance(data_dict["indicators"], dict)
        assert "rsi" in data_dict["indicators"]
