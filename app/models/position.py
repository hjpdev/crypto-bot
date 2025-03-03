from typing import Optional, List
from datetime import datetime
from decimal import Decimal
from enum import Enum
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    DateTime,
    Numeric,
    String,
    Text,
    Boolean,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship, Session

from app.models.base_model import BaseModel
from app.core.exceptions import ValidationError


class PositionType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"


class Position(BaseModel):
    """
    Model representing a cryptocurrency position.

    Attributes:
        cryptocurrency_id (int): Foreign key to Cryptocurrency
        symbol (str): Symbol of the cryptocurrency
        entry_timestamp (datetime): When the position was entered
        entry_price (Decimal): Entry price of the position
        size (Decimal): Size of the position in base currency
        position_type (Enum): LONG or SHORT
        stop_loss_price (Decimal): Stop loss price level
        take_profit_price (Decimal): Take profit price level
        status (Enum): Current status of the trade (OPEN/PARTIALLY_CLOSED/CLOSED)
        exit_timestamp (datetime, optional): When the position was fully closed
        exit_price (Decimal, optional): Final exit price of the position
        profit_loss (Decimal, optional): P&L in quote currency
        profit_loss_percentage (Decimal, optional): P&L as percentage
        strategy_used (str): Name of the strategy that triggered this trade
        notes (str, optional): Additional notes about the trade

    Relationships:
        cryptocurrency: Many-to-one relationship with Cryptocurrency
        partial_exits: One-to-many relationship with PartialExit
    """

    __tablename__ = "positions"

    cryptocurrency_id = Column(
        Integer, ForeignKey("cryptocurrencies.id"), nullable=False, index=True
    )
    symbol = Column(String(20), nullable=False, index=True)
    entry_timestamp = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    size = Column(Numeric(precision=18, scale=8), nullable=False)
    position_type = Column(SQLEnum(PositionType), nullable=False)
    stop_loss_price = Column(Numeric(precision=18, scale=8), nullable=False)
    take_profit_price = Column(Numeric(precision=18, scale=8), nullable=False)
    status = Column(SQLEnum(PositionStatus), nullable=False, default=PositionStatus.OPEN)
    exit_timestamp = Column(DateTime, nullable=True)
    exit_price = Column(Numeric(precision=18, scale=8), nullable=True)
    profit_loss = Column(Numeric(precision=18, scale=8), nullable=True)
    profit_loss_percentage = Column(
        Numeric(precision=10, scale=2), nullable=True
    )  # Stored as percentage (e.g., 5.25 for 5.25%)
    strategy_used = Column(String(100), nullable=False)
    notes = Column(Text, nullable=True)

    cryptocurrency = relationship("Cryptocurrency", back_populates="positions")
    partial_exits = relationship(
        "PartialExit", back_populates="position", cascade="all, delete-orphan"
    )

    def calculate_current_pl(self, current_price: Decimal) -> tuple[Decimal, Decimal]:
        """
        Calculate the current profit/loss based on the current market price.

        Args:
            current_price (Decimal): The current market price of the cryptocurrency

        Returns:
            tuple: (profit_loss, profit_loss_percentage)
        """
        if self.status == PositionStatus.CLOSED:
            return self.profit_loss, self.profit_loss_percentage

        remaining_size = self.size
        for partial_exit in self.partial_exits:
            remaining_size -= self.size * (partial_exit.exit_percentage / 100)

        if self.position_type == PositionType.LONG:
            pl = (current_price - self.entry_price) * remaining_size
            pl_percentage = ((current_price / self.entry_price) - 1) * 100
        else:
            pl = (self.entry_price - current_price) * remaining_size
            pl_percentage = ((self.entry_price / current_price) - 1) * 100

        return Decimal(pl), Decimal(pl_percentage)

    def apply_exit(
        self,
        price: Decimal,
        timestamp: datetime,
        full_exit: bool = False,
        exit_percentage: Optional[Decimal] = None,
    ) -> Optional["PartialExit"]:
        """
        Record a full or partial exit from this position.

        Args:
            price (Decimal): The exit price
            timestamp (datetime): When the exit occurred
            full_exit (bool): Whether this is a full exit
            exit_percentage (Decimal, optional): For partial exits, the percentage to exit

        Returns:
            PartialExit or None: If partial exit, returns the PartialExit instance

        Raises:
            ValidationError: If invalid exit parameters are provided
        """
        if self.status == PositionStatus.CLOSED:
            raise ValidationError("Cannot exit an already closed trade")

        if full_exit:
            self.exit_timestamp = timestamp
            self.exit_price = price
            self.status = PositionStatus.CLOSED

            remaining_size = self.size
            total_pl = Decimal(0)

            for partial_exit in self.partial_exits:
                partial_size = self.size * (partial_exit.exit_percentage / 100)
                remaining_size -= partial_size
                total_pl += partial_exit.profit_loss

            if self.position_type == PositionType.LONG:
                remaining_pl = (price - self.entry_price) * remaining_size
                pl_percentage = ((price / self.entry_price) - 1) * 100
            else:
                remaining_pl = (self.entry_price - price) * remaining_size
                pl_percentage = ((self.entry_price / price) - 1) * 100

            self.profit_loss = total_pl + remaining_pl
            self.profit_loss_percentage = pl_percentage

            return None
        else:
            if exit_percentage is None or exit_percentage <= 0 or exit_percentage >= 100:
                raise ValidationError(
                    "For partial exits, exit_percentage must be between 0 and 100"
                )

            exit_size = self.size * (exit_percentage / 100)

            if self.position_type == PositionType.LONG:
                pl = (price - self.entry_price) * exit_size
                pl_percentage = ((price / self.entry_price) - 1) * 100
            else:
                pl = (self.entry_price - price) * exit_size
                pl_percentage = ((self.entry_price / price) - 1) * 100

            partial_exit = PartialExit(
                position_id=self.id,
                exit_timestamp=timestamp,
                exit_price=price,
                exit_percentage=exit_percentage,
                profit_loss=pl,
                trailing_stop_activated=False,
            )

            self.status = PositionStatus.PARTIALLY_CLOSED

            return partial_exit

    def should_exit(self, current_price: Decimal) -> bool:
        if self.status == PositionStatus.CLOSED:
            return False

        if self.position_type == PositionType.LONG:
            if current_price <= self.stop_loss_price:
                return True
            if current_price >= self.take_profit_price:
                return True
        else:
            if current_price >= self.stop_loss_price:
                return True
            if current_price <= self.take_profit_price:
                return True

        return False

    @classmethod
    def get_open_positions(cls, session: Session) -> List["Position"]:
        """Get all open positions from the database."""
        return (
            session.query(cls)
            .filter(cls.status.in_([PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]))
            .all()
        )


class PartialExit(BaseModel):
    """
    Model representing a partial exit from a position.

    Attributes:
        position_id (int): Foreign key to Position
        exit_timestamp (datetime): When the partial exit occurred
        exit_price (Decimal): The price at which the partial exit occurred
        exit_percentage (Decimal): The percentage of the original position that was exited
        profit_loss (Decimal): P&L for this partial exit
        trailing_stop_activated (bool): Whether this exit was triggered by a trailing stop

    Relationships:
        position: Many-to-one relationship with Position
    """

    __tablename__ = "partial_exits"

    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False, index=True)
    exit_timestamp = Column(DateTime, nullable=False)
    exit_price = Column(Numeric(precision=18, scale=8), nullable=False)
    exit_percentage = Column(
        Numeric(precision=10, scale=2), nullable=False
    )  # Stored as percentage (e.g., 5.25 for 5.25%)
    profit_loss = Column(Numeric(precision=18, scale=8), nullable=False)
    trailing_stop_activated = Column(Boolean, nullable=False, default=False)

    position = relationship("Position", back_populates="partial_exits")
