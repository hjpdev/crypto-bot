from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Boolean, Date, Numeric, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.exc import SQLAlchemyError
from decimal import Decimal

from app.models.base_model import BaseModel
from app.core.exceptions import DatabaseError, ValidationError
from app.utils.logger import logger


class Cryptocurrency(BaseModel):
    """
    Model representing a cryptocurrency.

    Attributes:
        symbol (str): Trading symbol (e.g., "BTC/USD")
        name (str): Full name of the cryptocurrency (e.g., "Bitcoin")
        is_active (bool): Whether the cryptocurrency is actively traded
        market_cap (float): Market capitalization in USD
        avg_daily_volume (float): Average daily trading volume in USD
        exchange_specific_id (str): ID used by specific exchanges
        listing_date (date): Date when the cryptocurrency was first listed
    """

    __tablename__ = "cryptocurrencies"

    symbol = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    market_cap = Column(Numeric(precision=18, scale=2), nullable=True)
    avg_daily_volume = Column(Numeric(precision=18, scale=2), nullable=True)
    exchange_specific_id = Column(String(50), nullable=True)
    listing_date = Column(Date, nullable=True)

    # Define relationships - we'll only set up the relationship definitions here
    # and comment out the backref relationships until we update the other models
    # We'll add proper foreign keys when implementing or updating the related models

    # Relationships will be created when the related models are updated with foreign keys
    market_data = relationship(
        "OHLCV", back_populates="cryptocurrency", cascade="all, delete-orphan"
    )
    positions = relationship(
        "Position", back_populates="cryptocurrency", cascade="all, delete-orphan"
    )
    # market_snapshots =
    #   relationship(
    #       "MarketSnapshot", back_populates="cryptocurrency", cascade="all, delete-orphan"
    #   )

    __table_args__ = (UniqueConstraint("symbol", name="uix_crypto_symbol"),)

    def __repr__(self) -> str:
        """Return string representation of the cryptocurrency."""
        return (
            f"<Cryptocurrency(symbol='{self.symbol}', name='{self.name}', active={self.is_active})>"
        )

    @classmethod
    def get_by_symbol(cls, session, symbol: str) -> Optional["Cryptocurrency"]:
        try:
            return session.query(cls).filter(cls.symbol == symbol).first()
        except SQLAlchemyError as e:
            logger.error(
                f"Database error when fetching cryptocurrency by symbol {symbol}: {str(e)}"
            )
            raise DatabaseError(f"Error fetching cryptocurrency: {str(e)}") from e

    @classmethod
    def get_active(cls, session) -> List["Cryptocurrency"]:
        """Return all active cryptocurrencies."""
        try:
            return session.query(cls).filter(cls.is_active).all()
        except SQLAlchemyError as e:
            logger.error(f"Database error when fetching active cryptocurrencies: {str(e)}")
            raise DatabaseError(f"Error fetching active cryptocurrencies: {str(e)}") from e

    def update_market_data(self, session, data: Dict[str, Any]) -> None:
        """Update market data for specified cryptocurrency."""
        try:
            if (
                "market_cap" in data
                and not isinstance(data["market_cap"], (int, float, Decimal))
                and data["market_cap"] is not None
            ):
                raise ValidationError("Market cap must be a number")

            if (
                "avg_daily_volume" in data
                and not isinstance(data["avg_daily_volume"], (int, float, Decimal))
                and data["avg_daily_volume"] is not None
            ):
                raise ValidationError("Average daily volume must be a number")

            if "market_cap" in data:
                self.market_cap = data["market_cap"]

            if "avg_daily_volume" in data:
                self.avg_daily_volume = data["avg_daily_volume"]

            session.add(self)
            session.commit()

            logger.info(f"Updated market data for {self.symbol}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error when updating market data for {self.symbol}: {str(e)}")
            raise DatabaseError(f"Error updating market data: {str(e)}") from e
