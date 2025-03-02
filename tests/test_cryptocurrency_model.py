import pytest
from datetime import date, timedelta
from decimal import Decimal
from sqlalchemy.exc import IntegrityError

from app.models.cryptocurrency import Cryptocurrency
from app.core.exceptions import ValidationError


class TestCryptocurrencyModel:
    """Tests for the Cryptocurrency model."""

    @pytest.fixture
    def sample_crypto(self):
        """Create a sample cryptocurrency for testing."""
        return Cryptocurrency(
            symbol="BTC/USD",
            name="Bitcoin",
            market_cap=Decimal("800000000000.00"),
            avg_daily_volume=Decimal("30000000000.00"),
            exchange_specific_id="btc",
            listing_date=date.today() - timedelta(days=3000)
        )

    def test_create_cryptocurrency(self, db_session, sample_crypto):
        """Test creating a new cryptocurrency."""
        db_session.add(sample_crypto)
        db_session.commit()

        saved_crypto = db_session.query(Cryptocurrency).filter_by(symbol="BTC/USD").first()
        assert saved_crypto is not None
        assert saved_crypto.name == "Bitcoin"
        assert saved_crypto.is_active is True  # Default value
        assert saved_crypto.market_cap == Decimal("800000000000.00")

    def test_update_cryptocurrency(self, db_session, sample_crypto):
        """Test updating a cryptocurrency."""
        db_session.add(sample_crypto)
        db_session.commit()

        saved_crypto = db_session.query(Cryptocurrency).filter_by(symbol="BTC/USD").first()
        saved_crypto.name = "Bitcoin Updated"
        saved_crypto.is_active = False
        db_session.commit()

        updated_crypto = db_session.query(Cryptocurrency).filter_by(symbol="BTC/USD").first()
        assert updated_crypto.name == "Bitcoin Updated"
        assert updated_crypto.is_active is False

    def test_delete_cryptocurrency(self, db_session, sample_crypto):
        """Test deleting a cryptocurrency."""
        db_session.add(sample_crypto)
        db_session.commit()

        db_session.delete(sample_crypto)
        db_session.commit()

        deleted_crypto = db_session.query(Cryptocurrency).filter_by(symbol="BTC/USD").first()
        assert deleted_crypto is None

    def test_unique_symbol_constraint(self, db_session, sample_crypto):
        """Test that the symbol must be unique."""
        db_session.add(sample_crypto)
        db_session.commit()

        # Create another cryptocurrency with the same symbol
        duplicate_crypto = Cryptocurrency(
            symbol="BTC/USD",
            name="Bitcoin Duplicate"
        )

        db_session.add(duplicate_crypto)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_get_by_symbol(self, db_session, sample_crypto):
        """Test get_by_symbol class method."""
        db_session.add(sample_crypto)
        db_session.commit()

        retrieved_crypto = Cryptocurrency.get_by_symbol(db_session, "BTC/USD")
        assert retrieved_crypto is not None
        assert retrieved_crypto.name == "Bitcoin"

        # Test with non-existent symbol
        non_existent = Cryptocurrency.get_by_symbol(db_session, "NONEXISTENT")
        assert non_existent is None

    def test_get_active(self, db_session):
        """Test get_active class method."""
        # Create multiple cryptocurrencies with different active status
        crypto1 = Cryptocurrency(symbol="BTC/USD", name="Bitcoin", is_active=True)
        crypto2 = Cryptocurrency(symbol="ETH/USD", name="Ethereum", is_active=True)
        crypto3 = Cryptocurrency(symbol="XRP/USD", name="Ripple", is_active=False)

        db_session.add_all([crypto1, crypto2, crypto3])
        db_session.commit()

        active_cryptos = Cryptocurrency.get_active(db_session)
        assert len(active_cryptos) == 2
        assert all(crypto.is_active for crypto in active_cryptos)
        assert "BTC/USD" in [crypto.symbol for crypto in active_cryptos]
        assert "ETH/USD" in [crypto.symbol for crypto in active_cryptos]
        assert "XRP/USD" not in [crypto.symbol for crypto in active_cryptos]

    def test_update_market_data(self, db_session, sample_crypto):
        """Test update_market_data method."""
        db_session.add(sample_crypto)
        db_session.commit()

        # Update market data
        new_data = {
            "market_cap": Decimal("850000000000.00"),
            "avg_daily_volume": Decimal("35000000000.00")
        }

        sample_crypto.update_market_data(db_session, new_data)

        # Verify the update
        updated_crypto = db_session.query(Cryptocurrency).filter_by(symbol="BTC/USD").first()
        assert updated_crypto.market_cap == Decimal("850000000000.00")
        assert updated_crypto.avg_daily_volume == Decimal("35000000000.00")

        # Test with invalid data
        with pytest.raises(ValidationError):
            sample_crypto.update_market_data(db_session, {"market_cap": "invalid"})

    @pytest.mark.parametrize("test_data,expected_error", [
        ({"market_cap": "invalid"}, ValidationError),
        ({"avg_daily_volume": "invalid"}, ValidationError),
    ])
    def test_update_market_data_validation(
        self, db_session, sample_crypto, test_data, expected_error
    ):
        """Test validation in update_market_data method."""
        db_session.add(sample_crypto)
        db_session.commit()

        with pytest.raises(expected_error):
            sample_crypto.update_market_data(db_session, test_data)

    @pytest.fixture
    def db_session(self):
        """
        Fixture to provide a database session.

        This is a placeholder that would need to be implemented according to your testing setup.
        It should provide a SQLAlchemy session connected to a test database.
        """
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.models.base_model import BaseModel

        # Create an in-memory SQLite database for testing
        engine = create_engine('sqlite:///:memory:')
        BaseModel.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        yield session

        session.close()
        BaseModel.metadata.drop_all(engine)
