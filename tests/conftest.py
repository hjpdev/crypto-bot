import pytest
from unittest import mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
from decimal import Decimal
import time

# Create a memory-only engine and base for testing
TestingBase = declarative_base()


# Mock models that mirror the real ones but work with our in-memory database
class MockBaseModel(TestingBase):
    __abstract__ = True

    from sqlalchemy import Column, Integer, DateTime

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        columns = [c.name for c in self.__table__.columns]
        values = [getattr(self, c) for c in columns]
        params = ", ".join(f"{c}={v!r}" for c, v in zip(columns, values))
        return f"{self.__class__.__name__}({params})"


class MockCryptocurrency(MockBaseModel):
    __tablename__ = "cryptocurrencies"

    from sqlalchemy import Column, String, Boolean, Numeric, Index, UniqueConstraint, DateTime
    from sqlalchemy.orm import relationship

    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    market_cap = Column(Numeric(precision=24, scale=2), nullable=True)
    avg_daily_volume = Column(Numeric(precision=24, scale=2), nullable=True)
    exchange_specific_id = Column(String(100), nullable=True)
    listing_date = Column(DateTime, nullable=True)

    # Define the relationship to OHLCV
    market_data = relationship(
        "MockOHLCV",
        back_populates="cryptocurrency",
        cascade="all, delete-orphan"
    )

    # Define relationship to MarketSnapshot
    market_snapshots = relationship(
        "MockMarketSnapshot",
        back_populates="cryptocurrency",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("symbol", name="uix_crypto_symbol"),
        Index("ix_cryptocurrencies_active", "is_active"),
    )

    @classmethod
    def get_by_symbol(cls, session, symbol):
        return session.query(cls).filter(cls.symbol == symbol).first()

    @classmethod
    def get_active(cls, session):
        return session.query(cls).filter(cls.is_active).all()

    def update_market_data(self, session, market_data):
        # Mock of the original method that doesn't actually need to validate
        return True


class MockOHLCV(MockBaseModel):
    __tablename__ = "ohlcv"

    from sqlalchemy import (
        Column, String, DateTime, Index, UniqueConstraint, JSON, ForeignKey, Numeric, Integer
    )
    from sqlalchemy.orm import relationship

    cryptocurrency_id = Column(
        Integer,
        ForeignKey("cryptocurrencies.id"),
        nullable=False,
        index=True
    )
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Numeric(precision=18, scale=8), nullable=False)
    high = Column(Numeric(precision=18, scale=8), nullable=False)
    low = Column(Numeric(precision=18, scale=8), nullable=False)
    close = Column(Numeric(precision=18, scale=8), nullable=False)
    volume = Column(Numeric(precision=24, scale=8), nullable=False)
    indicators = Column(JSON, nullable=True)

    cryptocurrency = relationship("MockCryptocurrency", back_populates="market_data")

    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "timeframe", "timestamp", name="uix_ohlcv"),
        Index("ix_ohlcv_query", "exchange", "symbol", "timeframe", "timestamp"),
    )

    @property
    def as_dict(self):
        # Simplified version of the as_dict property
        return {
            "id": self.id,
            "cryptocurrency_id": self.cryptocurrency_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "open": float(self.open) if self.open else None,
            "high": float(self.high) if self.high else None,
            "low": float(self.low) if self.low else None,
            "close": float(self.close) if self.close else None,
            "volume": float(self.volume) if self.volume else None,
            "indicators": self.indicators or {},
        }

    @classmethod
    def get_latest(cls, session, symbol, timeframe="1h", limit=100):
        from sqlalchemy import desc
        return (
            session.query(cls)
            .filter(cls.symbol == symbol)
            .filter(cls.timeframe == timeframe)
            .order_by(desc(cls.timestamp))
            .limit(limit)
            .all()
        )

    @classmethod
    def get_range(cls, session, symbol, start, end, timeframe="1h"):
        # Ensure timezone information is preserved
        start = cls.ensure_timezone(start)
        end = cls.ensure_timezone(end)

        return (
            session.query(cls)
            .filter(cls.symbol == symbol)
            .filter(cls.timeframe == timeframe)
            .filter(cls.timestamp >= start)
            .filter(cls.timestamp <= end)
            .order_by(cls.timestamp)
            .all()
        )

    @staticmethod
    def validate_ohlcv_data(open_price, high_price, low_price, close_price, volume):
        # Mock of the validation method
        from app.core.exceptions import ValidationError

        if high_price < low_price:
            raise ValidationError("High price cannot be lower than low price")

        if any(p < 0 for p in [open_price, high_price, low_price, close_price, volume]):
            raise ValidationError("Price values cannot be negative")

        return True

    @classmethod
    def ensure_timezone(cls, dt):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def update_indicators(self, session, indicators):
        """
        Update the indicators JSON field with calculated technical indicators.
        """
        if self.indicators:
            # Update existing indicators
            current_indicators = self.indicators.copy()
            current_indicators.update(indicators)
            self.indicators = current_indicators
        else:
            # Set new indicators
            self.indicators = indicators

        session.add(self)
        session.commit()
        return True


class MockMarketSnapshot(MockBaseModel):
    __tablename__ = "market_snapshots"

    from sqlalchemy import (
        Column, String, DateTime, Index, UniqueConstraint, JSON, ForeignKey, Numeric, Integer
    )
    from sqlalchemy.orm import relationship

    cryptocurrency_id = Column(
        Integer,
        ForeignKey("cryptocurrencies.id"),
        nullable=False,
        index=True
    )
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ohlcv = Column(JSON, nullable=False)
    indicators = Column(JSON, nullable=False)
    order_book = Column(JSON, nullable=False)
    trading_volume = Column(Numeric(precision=24, scale=8), nullable=False)
    market_sentiment = Column(Numeric(precision=10, scale=2), nullable=True)
    correlation_btc = Column(Numeric(precision=5, scale=4), nullable=True)

    cryptocurrency = relationship("MockCryptocurrency", back_populates="market_snapshots")

    __table_args__ = (
        UniqueConstraint(
            "cryptocurrency_id", "timestamp", name="uix_market_snapshot_crypto_time"
        ),
    )

    @property
    def as_dict(self):
        """Simplified version of the as_dict property for testing"""
        return {
            "id": self.id,
            "cryptocurrency_id": self.cryptocurrency_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "ohlcv": self.ohlcv,
            "indicators": self.indicators,
            "order_book": self.order_book,
            "trading_volume": float(self.trading_volume) if self.trading_volume else None,
            "market_sentiment": float(self.market_sentiment) if self.market_sentiment else None,
            "correlation_btc": float(self.correlation_btc) if self.correlation_btc else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def get_latest(cls, session, cryptocurrency_id):
        """Mock of get_latest for testing"""
        return (
            session.query(cls)
            .filter(cls.cryptocurrency_id == cryptocurrency_id)
            .order_by(cls.timestamp.desc())
            .first()
        )

    @classmethod
    def get_range(cls, session, cryptocurrency_id, start, end):
        """Mock of get_range for testing"""
        # Ensure datetimes have timezone info if not provided
        start = cls.ensure_timezone(start)
        end = cls.ensure_timezone(end)

        return (
            session.query(cls)
            .filter(cls.cryptocurrency_id == cryptocurrency_id)
            .filter(cls.timestamp >= start)
            .filter(cls.timestamp <= end)
            .order_by(cls.timestamp)
            .all()
        )

    @classmethod
    def get_with_specific_indicator(cls, session, cryptocurrency_id, indicator):
        """Mock of get_with_specific_indicator for testing"""
        # For testing, we'll just check if the indicator exists as a key at the top level
        # This is a simplification for the test environment
        from sqlalchemy import text
        return (
            session.query(cls)
            .filter(cls.cryptocurrency_id == cryptocurrency_id)
            .filter(text(f"json_extract(indicators, '$.{indicator}') IS NOT NULL"))
            .order_by(cls.timestamp.desc())
            .all()
        )

    @staticmethod
    def ensure_timezone(dt):
        """Ensure datetime has timezone info"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


@pytest.fixture(scope="session")
def db_url():
    """Provide an in-memory SQLite database URL for testing."""
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def db_engine(db_url):
    """Create a database engine for testing using SQLite in-memory."""
    engine = create_engine(db_url)

    # Create all tables
    TestingBase.metadata.create_all(engine)

    # Set up mocks to replace the real models
    patch_base = mock.patch('app.models.base_model.Base', TestingBase)
    patch_crypto = mock.patch('app.models.cryptocurrency.Cryptocurrency', MockCryptocurrency)
    patch_ohlcv = mock.patch('app.models.ohlcv.OHLCV', MockOHLCV)
    patch_market_snapshot = mock.patch(
        'app.models.market_snapshot.MarketSnapshot', MockMarketSnapshot
    )

    # Start the patches
    patch_base.start()
    patch_crypto.start()
    patch_ohlcv.start()
    patch_market_snapshot.start()

    yield engine

    # Clean up
    TestingBase.metadata.drop_all(engine)

    # Stop the patches
    patch_base.stop()
    patch_crypto.stop()
    patch_ohlcv.stop()
    patch_market_snapshot.stop()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Provide a database session for testing."""
    Session = sessionmaker(bind=db_engine)
    session = Session()

    # Patch any imports of the actual models to use our mock models instead
    with mock.patch('app.models.cryptocurrency.Cryptocurrency', MockCryptocurrency), \
         mock.patch('app.models.ohlcv.OHLCV', MockOHLCV), \
         mock.patch('app.models.market_snapshot.MarketSnapshot', MockMarketSnapshot):
        yield session

    session.rollback()
    session.close()


@pytest.fixture
def sample_crypto(db_session):
    """Create a sample cryptocurrency for testing."""
    # Using a timestamp to create a unique symbol for each test
    timestamp = int(time.time() * 1000)

    crypto = MockCryptocurrency(
        symbol=f"BTC/USD_{timestamp}",
        name="Bitcoin",
        is_active=True,
        market_cap=Decimal("800000000000.00"),
        avg_daily_volume=Decimal("30000000000.00"),
        exchange_specific_id="btc",
    )
    db_session.add(crypto)
    db_session.commit()
    return crypto


@pytest.fixture
def sample_ohlcv(db_session, sample_crypto):
    """Create a sample OHLCV record for testing."""
    timestamp = datetime.now(timezone.utc)
    ohlcv = MockOHLCV(
        cryptocurrency_id=sample_crypto.id,
        exchange="binance",
        symbol=sample_crypto.symbol,
        timeframe="1h",
        timestamp=timestamp,
        open=Decimal("50000.00"),
        high=Decimal("52000.00"),
        low=Decimal("49000.00"),
        close=Decimal("51000.00"),
        volume=Decimal("1000.50"),
        indicators={"rsi": 65.5, "macd": {"signal": 0.5}}
    )
    db_session.add(ohlcv)
    db_session.commit()
    return ohlcv
