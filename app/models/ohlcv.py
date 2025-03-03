from typing import List, Dict, Any, Union
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Index,
    UniqueConstraint,
    JSON,
    ForeignKey,
    Numeric,
    desc,
    Integer,
)
from sqlalchemy.orm import relationship, Session
from sqlalchemy.exc import SQLAlchemyError
from app.models.base_model import BaseModel
from app.core.exceptions import DatabaseError, ValidationError
from app.utils.logger import logger


class OHLCV(BaseModel):
    """
    Model for storing OHLCV (Open, High, Low, Close, Volume) market data.

    Attributes:
        cryptocurrency_id (int): Foreign key to the cryptocurrency
        exchange (str): Exchange name (e.g., "binance", "coinbase")
        symbol (str): Trading symbol (e.g., "BTC/USD")
        timeframe (str): Timeframe of the data (e.g., "1m", "1h", "1d")
        timestamp (datetime): Timestamp of the data point in UTC
        open (Decimal): Opening price
        high (Decimal): Highest price during the period
        low (Decimal): Lowest price during the period
        close (Decimal): Closing price
        volume (Decimal): Trading volume
        indicators (dict): JSON column storing calculated indicators
    """

    __tablename__ = "ohlcv"

    cryptocurrency_id = Column(
        Integer, ForeignKey("cryptocurrencies.id"), nullable=False, index=True
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

    cryptocurrency = relationship("Cryptocurrency", back_populates="market_data")

    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "timeframe", "timestamp", name="uix_ohlcv"),
        Index("ix_ohlcv_query", "exchange", "symbol", "timeframe", "timestamp"),
    )

    def __repr__(self) -> str:
        """Return string representation of the OHLCV data."""
        return (
            f"<OHLCV(symbol='{self.symbol}', exchange='{self.exchange}', "
            f"timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"
        )

    @property
    def as_dict(self) -> Dict[str, Any]:
        """Return the OHLCV data as a dictionary."""
        return {
            "id": self.id,
            "cryptocurrency_id": self.cryptocurrency_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "indicators": self.indicators or {},
        }

    @classmethod
    def get_latest(
        cls, session: Session, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> List["OHLCV"]:
        """Get the most recent OHLCV data for a cryptocurrency."""
        try:
            return (
                session.query(cls)
                .filter(cls.symbol == symbol)
                .filter(cls.timeframe == timeframe)
                .order_by(desc(cls.timestamp))
                .limit(limit)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(
                f"Database error when fetching latest OHLCV data for crypto ID {symbol}: {str(e)}"
            )
            raise DatabaseError(f"Error fetching latest OHLCV data: {str(e)}") from e

    @classmethod
    def get_range(
        cls,
        session: Session,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1h",
    ) -> List["OHLCV"]:
        """
        Get OHLCV data for a cryptocurrency within a specified time range.

        Args:
            session: SQLAlchemy session
            symbol: Symbol of the cryptocurrency
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            timeframe: Timeframe of the data (e.g., "1m", "1h", "1d")

        Returns:
            List of OHLCV objects ordered by timestamp (ascending)

        Raises:
            DatabaseError: If there was an error accessing the database
            ValidationError: If end time is before start time
        """
        if end < start:
            raise ValidationError("End time must be after start time")

        # Ensure timezone information is preserved
        start = cls.ensure_timezone(start)
        end = cls.ensure_timezone(end)

        try:
            return (
                session.query(cls)
                .filter(cls.symbol == symbol)
                .filter(cls.timeframe == timeframe)
                .filter(cls.timestamp >= start)
                .filter(cls.timestamp <= end)
                .order_by(cls.timestamp)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(
                f"Database error when fetching OHLCV range for crypto ID {symbol}: {str(e)}"
            )
            raise DatabaseError(f"Error fetching OHLCV data range: {str(e)}") from e

    def update_indicators(self, session: Session, indicators: Dict[str, Any]) -> None:
        """
        Update the indicators JSON field with calculated technical indicators.

        Args:
            session: SQLAlchemy session
            indicators: Dictionary of indicator values to store

        Raises:
            DatabaseError: If there was an error updating the database
        """
        try:
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

            logger.debug(f"Updated indicators for OHLCV ID {self.id}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(
                f"Database error when updating indicators for OHLCV ID {self.id}: {str(e)}"
            )
            raise DatabaseError(f"Error updating indicators: {str(e)}") from e

    @staticmethod
    def validate_ohlcv_data(
        open_price: Union[float, int],
        high_price: Union[float, int],
        low_price: Union[float, int],
        close_price: Union[float, int],
        volume: Union[float, int],
    ) -> None:
        """
        Validate OHLCV data to ensure it meets logical requirements.

        Args:
            open_price: Opening price
            high_price: Highest price
            low_price: Lowest price
            close_price: Closing price
            volume: Trading volume

        Raises:
            ValidationError: If any validation checks fail
        """
        if not all(
            isinstance(p, (float, int))
            for p in [open_price, high_price, low_price, close_price, volume]
        ):
            raise ValidationError("All price and volume values must be numbers")

        if high_price < low_price:
            raise ValidationError(
                "High price cannot be lower than low price - data is inconsistent"
            )

        if any(p < 0 for p in [open_price, high_price, low_price, close_price]):
            raise ValidationError("Price values cannot be negative - data is inconsistent")

        if volume < 0:
            raise ValidationError("Volume cannot be negative - data is inconsistent")

        # Check for logical consistency in price values
        if open_price > high_price:
            raise ValidationError(
                "Open price cannot be higher than high price - data is inconsistent"
            )

        if close_price > high_price:
            raise ValidationError(
                "Close price cannot be higher than high price - data is inconsistent"
            )

        if open_price < low_price:
            raise ValidationError(
                "Open price cannot be lower than low price - data is inconsistent"
            )

        if close_price < low_price:
            raise ValidationError(
                "Close price cannot be lower than low price - data is inconsistent"
            )

    @classmethod
    def ensure_timezone(cls, dt):
        """
        Ensure a datetime has timezone information (UTC).

        Args:
            dt: A datetime object

        Returns:
            A datetime with UTC timezone information
        """
        if dt is None:
            return None

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        else:
            return dt.astimezone(timezone.utc)
