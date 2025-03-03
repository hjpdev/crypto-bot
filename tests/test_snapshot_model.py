import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from sqlalchemy.exc import IntegrityError

from app.models.market_snapshot import MarketSnapshot
from app.models.cryptocurrency import Cryptocurrency


class TestMarketSnapshotModel:
    """Tests for the MarketSnapshot model."""

    @pytest.fixture(autouse=True)
    def cleanup_snapshots(self, db_session):
        """Automatically clean up snapshots between tests."""
        yield
        # After each test, delete all snapshots
        db_session.query(MarketSnapshot).delete()
        db_session.commit()

    @pytest.fixture
    def sample_crypto(self, db_session):
        """Create a sample cryptocurrency for testing."""
        # First check if cryptocurrency with this symbol already exists
        existing_crypto = db_session.query(Cryptocurrency).filter_by(symbol="BTC/USD").first()

        if existing_crypto:
            return existing_crypto

        # If not, create a new one
        crypto = Cryptocurrency(
            symbol="BTC/USD",
            name="Bitcoin",
            market_cap=Decimal("800000000000.00"),
            avg_daily_volume=Decimal("30000000000.00"),
            exchange_specific_id="btc",
            listing_date=datetime.now(timezone.utc).date(),
        )
        db_session.add(crypto)
        db_session.commit()
        return crypto

    @pytest.fixture
    def sample_snapshot(self, sample_crypto):
        """Create a sample market snapshot for testing."""
        now = datetime.now(timezone.utc)
        return MarketSnapshot(
            cryptocurrency_id=sample_crypto.id,
            symbol=sample_crypto.symbol,
            timestamp=now,
            ohlcv={
                "1h": {
                    "open": 50000.0,
                    "high": 51000.0,
                    "low": 49500.0,
                    "close": 50500.0,
                    "volume": 1000.0,
                },
                "4h": {
                    "open": 49000.0,
                    "high": 52000.0,
                    "low": 48500.0,
                    "close": 50500.0,
                    "volume": 4000.0,
                },
            },
            indicators={
                "rsi_14": 65.5,
                "ma_50": 48000.0,
                "ma_200": 45000.0,
            },
            order_book={
                "bids": [
                    [50000.0, 1.5],
                    [49900.0, 2.3],
                    [49800.0, 3.1],
                ],
                "asks": [
                    [50100.0, 1.2],
                    [50200.0, 2.5],
                    [50300.0, 3.0],
                ],
                "timestamp": now.isoformat(),
            },
            trading_volume=Decimal("1200000000.00"),
            market_sentiment=Decimal("65.5"),
            correlation_btc=Decimal("1.0000"),
        )

    def test_create_market_snapshot(self, db_session, sample_snapshot):
        """Test creating a new market snapshot."""
        db_session.add(sample_snapshot)
        db_session.commit()

        saved_snapshot = (
            db_session.query(MarketSnapshot)
            .filter_by(cryptocurrency_id=sample_snapshot.cryptocurrency_id)
            .first()
        )
        assert saved_snapshot is not None
        assert "BTC/USD" in saved_snapshot.symbol
        assert saved_snapshot.trading_volume == Decimal("1200000000.00")
        assert saved_snapshot.market_sentiment == Decimal("65.5")
        assert saved_snapshot.correlation_btc == Decimal("1.0000")

        # JSON field tests
        assert "rsi_14" in saved_snapshot.indicators
        assert saved_snapshot.indicators["rsi_14"] == 65.5
        assert "1h" in saved_snapshot.ohlcv
        assert saved_snapshot.ohlcv["1h"]["close"] == 50500.0
        assert "bids" in saved_snapshot.order_book
        assert len(saved_snapshot.order_book["bids"]) == 3

    def test_relationship_with_cryptocurrency(self, db_session, sample_crypto, sample_snapshot):
        """Test relationship between MarketSnapshot and Cryptocurrency."""
        db_session.add(sample_snapshot)
        db_session.commit()

        # Test access from snapshot to cryptocurrency
        snapshot = db_session.query(MarketSnapshot).first()
        assert snapshot.cryptocurrency is not None
        assert snapshot.cryptocurrency.id == sample_crypto.id
        assert snapshot.cryptocurrency.symbol == sample_crypto.symbol

        # Test access from cryptocurrency to snapshots
        crypto = db_session.query(Cryptocurrency).filter_by(id=sample_crypto.id).first()
        assert crypto.market_snapshots is not None
        assert len(crypto.market_snapshots) == 1
        assert crypto.market_snapshots[0].id == snapshot.id

    def test_json_field_functionality(self, db_session, sample_snapshot):
        """Test JSON field functionality."""
        db_session.add(sample_snapshot)
        db_session.commit()

        # Store the ID so we can query for this specific snapshot
        snapshot_id = sample_snapshot.id

        # Retrieve the snapshot
        snapshot = db_session.query(MarketSnapshot).filter_by(id=snapshot_id).first()

        # Test updating a JSON field by creating a new dictionary with added data
        updated_indicators = snapshot.indicators.copy()
        updated_indicators["macd"] = {"signal": 0.25, "histogram": 0.1, "macd": 0.35}
        snapshot.indicators = updated_indicators
        db_session.commit()

        # Retrieve again and check the update
        updated_snapshot = db_session.query(MarketSnapshot).filter_by(id=snapshot_id).first()
        assert "macd" in updated_snapshot.indicators
        assert updated_snapshot.indicators["macd"]["signal"] == 0.25

        # Test adding a new timeframe to OHLCV
        updated_ohlcv = snapshot.ohlcv.copy()
        updated_ohlcv["15m"] = {
            "open": 50200.0,
            "high": 50300.0,
            "low": 50100.0,
            "close": 50250.0,
            "volume": 500.0,
        }
        snapshot.ohlcv = updated_ohlcv
        db_session.commit()

        # Retrieve again and check the update
        updated_snapshot = db_session.query(MarketSnapshot).filter_by(id=snapshot_id).first()
        assert "15m" in updated_snapshot.ohlcv
        assert updated_snapshot.ohlcv["15m"]["close"] == 50250.0

    def test_unique_constraint(self, db_session, sample_crypto, sample_snapshot):
        """Test unique constraint on cryptocurrency_id and timestamp."""
        db_session.add(sample_snapshot)
        db_session.commit()

        # Try to create another snapshot with the same cryptocurrency_id and timestamp
        duplicate_snapshot = MarketSnapshot(
            cryptocurrency_id=sample_snapshot.cryptocurrency_id,
            symbol=sample_snapshot.symbol,
            timestamp=sample_snapshot.timestamp,
            ohlcv={
                "1h": {
                    "open": 50000.0,
                    "high": 51000.0,
                    "low": 49000.0,
                    "close": 50500.0,
                    "volume": 1000.0,
                }
            },
            indicators={"rsi": 65},
            order_book={"bids": [], "asks": []},
            trading_volume=Decimal("1000000.00"),
        )

        # Should raise an integrity error due to unique constraint
        with pytest.raises(IntegrityError):
            db_session.add(duplicate_snapshot)
            db_session.commit()

        # Roll back the failed transaction
        db_session.rollback()

    def test_get_latest(self, db_session, sample_crypto, sample_snapshot):
        """Test get_latest method."""
        # Add first snapshot
        db_session.add(sample_snapshot)
        db_session.commit()

        # Add newer snapshot
        now = datetime.now(timezone.utc)
        newer_snapshot = MarketSnapshot(
            cryptocurrency_id=sample_crypto.id,
            symbol=sample_crypto.symbol,
            timestamp=now + timedelta(hours=1),
            ohlcv={
                "1h": {
                    "open": 50500.0,
                    "high": 51500.0,
                    "low": 50000.0,
                    "close": 51000.0,
                    "volume": 1100.0,
                }
            },
            indicators={"rsi_14": 68.0},
            order_book={"bids": [], "asks": []},
            trading_volume=Decimal("1300000000.00"),
        )
        db_session.add(newer_snapshot)
        db_session.commit()

        # Test get_latest returns the most recent snapshot
        latest = MarketSnapshot.get_latest(db_session, sample_crypto.id)
        assert latest is not None
        assert latest.timestamp > sample_snapshot.timestamp
        assert latest.trading_volume == Decimal("1300000000.00")

    def test_get_range(self, db_session, sample_crypto):
        """Test get_range method."""
        base_time = datetime.now(timezone.utc)

        # Create snapshots with different timestamps
        snapshots = []
        for i in range(5):
            snapshot = MarketSnapshot(
                cryptocurrency_id=sample_crypto.id,
                symbol=sample_crypto.symbol,
                timestamp=base_time + timedelta(hours=i),
                ohlcv={
                    "1h": {
                        "open": 50000.0 + i*100,
                        "high": 51000.0 + i*100,
                        "low": 49000.0 + i*100,
                        "close": 50500.0 + i*100,
                        "volume": 1000.0 + i*10,
                    },
                },
                indicators={"rsi_14": 65.0 + i},
                order_book={"bids": [], "asks": []},
                trading_volume=Decimal(f"{1200000000.00 + i*10000000}"),
            )
            snapshots.append(snapshot)
            db_session.add(snapshot)

        db_session.commit()

        # Test get_range with full range
        all_snapshots = MarketSnapshot.get_range(
            db_session,
            sample_crypto.id,
            base_time,
            base_time + timedelta(hours=4)
        )
        assert len(all_snapshots) == 5

        # Test get_range with partial range
        partial_snapshots = MarketSnapshot.get_range(
            db_session,
            sample_crypto.id,
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=3)
        )
        assert len(partial_snapshots) == 3

        # Test range ordering
        assert partial_snapshots[0].timestamp < partial_snapshots[1].timestamp
        assert partial_snapshots[1].timestamp < partial_snapshots[2].timestamp

    def test_as_dict_property(self, db_session, sample_snapshot):
        """Test the as_dict property."""
        db_session.add(sample_snapshot)
        db_session.commit()

        snapshot = db_session.query(MarketSnapshot).first()
        snapshot_dict = snapshot.as_dict

        # Check basic properties
        assert snapshot_dict["id"] == snapshot.id
        assert snapshot_dict["cryptocurrency_id"] == snapshot.cryptocurrency_id
        assert snapshot_dict["symbol"] == snapshot.symbol

        # Check JSON fields
        assert snapshot_dict["ohlcv"] == snapshot.ohlcv
        assert snapshot_dict["indicators"] == snapshot.indicators
        assert snapshot_dict["order_book"] == snapshot.order_book

        # Check numeric fields are properly converted to float
        assert isinstance(snapshot_dict["trading_volume"], float)
        assert isinstance(snapshot_dict["market_sentiment"], float)
        assert isinstance(snapshot_dict["correlation_btc"], float)

        # Check timestamp is converted to ISO format string
        assert isinstance(snapshot_dict["timestamp"], str)
