from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    DateTime,
    UniqueConstraint,
    JSON,
    ForeignKey,
    Numeric,
    Integer,
)
from sqlalchemy.orm import relationship, Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.base_model import BaseModel
from app.core.exceptions import DatabaseError
from app.utils.logger import logger


class MarketSnapshot(BaseModel):
    """
    Model for storing comprehensive snapshots of cryptocurrency market data.

    Attributes:
        cryptocurrency_id (int): Foreign key to the cryptocurrency
        symbol (str): Trading symbol (e.g., "BTC/USD")
        timestamp (datetime): Timestamp of the snapshot in UTC
        ohlcv (dict): JSON column storing OHLCV data for multiple timeframes
        indicators (dict): JSON column storing calculated indicators
        order_book (dict): JSON column storing order book snapshot
        trading_volume (Decimal): Trading volume in the last 24 hours
        market_sentiment (Decimal): Market sentiment score, nullable
        correlation_btc (Decimal): Correlation with Bitcoin, nullable
    """

    __tablename__ = "market_snapshots"

    cryptocurrency_id = Column(
        Integer, ForeignKey("cryptocurrencies.id"), nullable=False, index=True
    )
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ohlcv = Column(JSON, nullable=False)
    indicators = Column(JSON, nullable=False)
    order_book = Column(JSON, nullable=False)
    trading_volume = Column(Numeric(precision=24, scale=8), nullable=False)
    market_sentiment = Column(Numeric(precision=10, scale=2), nullable=True)
    correlation_btc = Column(Numeric(precision=5, scale=4), nullable=True)

    # Relationship to Cryptocurrency (many-to-one)
    cryptocurrency = relationship("Cryptocurrency", back_populates="market_snapshots")

    __table_args__ = (
        UniqueConstraint("cryptocurrency_id", "timestamp", name="uix_market_snapshot_crypto_time"),
    )

    def __repr__(self) -> str:
        """Return string representation of the market snapshot."""
        return (
            f"<MarketSnapshot(symbol='{self.symbol}', "
            f"timestamp='{self.timestamp}', "
            f"cryptocurrency_id={self.cryptocurrency_id})>"
        )

    @property
    def as_dict(self) -> Dict[str, Any]:
        """Convert the snapshot to a dictionary format."""
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
    def get_latest(cls, session: Session, cryptocurrency_id: int) -> Optional["MarketSnapshot"]:
        """Get the most recent market snapshot for a specific cryptocurrency."""
        try:
            return (
                session.query(cls)
                .filter(cls.cryptocurrency_id == cryptocurrency_id)
                .order_by(cls.timestamp.desc())
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(
                f"Database error fetching latest market snapshot id: {cryptocurrency_id}: {str(e)}"
            )
            raise DatabaseError(f"Error fetching latest market snapshot: {str(e)}") from e

    @classmethod
    def get_range(
        cls, session: Session, cryptocurrency_id: int, start: datetime, end: datetime
    ) -> List["MarketSnapshot"]:
        """
        Get market snapshots for a cryptocurrency within a specified time range.

        Args:
            session: SQLAlchemy session
            cryptocurrency_id: ID of the cryptocurrency
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of MarketSnapshot objects

        Raises:
            DatabaseError: If there's an error querying the database
        """
        try:
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
        except SQLAlchemyError as e:
            logger.error(
                f"Database error fetching market snapshots, id: {cryptocurrency_id}: {str(e)}"
            )
            raise DatabaseError(f"Error fetching market snapshots: {str(e)}") from e

    @classmethod
    def get_with_specific_indicator(
        cls, session: Session, cryptocurrency_id: int, indicator: str
    ) -> List["MarketSnapshot"]:
        """
        Get market snapshots containing a specific indicator.

        Args:
            session: SQLAlchemy session
            cryptocurrency_id: ID of the cryptocurrency
            indicator: Name of the indicator to filter by

        Returns:
            List of MarketSnapshot objects containing the specified indicator

        Raises:
            DatabaseError: If there's an error querying the database
        """
        try:
            # PostgreSQL JSON containment operator for checking if indicator exists in the JSON
            # This assumes PostgreSQL as the database, may need adjustment for other databases
            return (
                session.query(cls)
                .filter(cls.cryptocurrency_id == cryptocurrency_id)
                .filter(cls.indicators.has_key(indicator))
                .order_by(cls.timestamp.desc())
                .all()
            )
        except SQLAlchemyError as e:
            logger.error(
                f"Database error when fetching snapshots with indicator {indicator}: {str(e)}"
            )
            raise DatabaseError(f"Error fetching snapshots with indicator: {str(e)}") from e

    @staticmethod
    def ensure_timezone(dt: datetime) -> datetime:
        """Ensure datetime has timezone information, using UTC if none is specified."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def update_indicators(self, session: Session, indicators: Dict[str, Any]) -> None:
        """Update the indicators for this market snapshot."""
        try:
            current_indicators = self.indicators or {}
            updated_indicators = {**current_indicators, **indicators}

            self.indicators = updated_indicators
            session.add(self)
            session.commit()

            logger.info(f"Updated indicators for {self.symbol} snapshot at {self.timestamp}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error when updating indicators: {str(e)}")
            raise DatabaseError(f"Error updating indicators: {str(e)}") from e
