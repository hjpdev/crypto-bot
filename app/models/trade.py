from sqlalchemy import Column, String, Float, DateTime, Index, JSON, Integer

from app.models.base_model import BaseModel


class Trade(BaseModel):
    """
    Model for storing individual trades and associated market conditions.
    """

    __tablename__ = "trades"

    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    position_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    side = Column(String(10), nullable=False, index=True)  # 'buy' or 'sell'
    trade_id = Column(String(100), nullable=True, unique=True, index=True)

    # Market conditions at the time of the trade
    market_price = Column(Float, nullable=True)  # Market price at time of trade
    spread = Column(Float, nullable=True)  # Bid-ask spread at time of trade
    volume_24h = Column(Float, nullable=True)  # 24-hour volume at time of trade

    # Technical indicators at the time of trade (optional)
    indicators = Column(JSON, nullable=True)  # Store indicators like RSI, MACD, etc.

    # Strategy information
    strategy = Column(String(50), nullable=True, index=True)  # Which strategy triggered this trade
    strategy_parameters = Column(JSON, nullable=True)  # Parameters used by the strategy

    # Trade performance tracking
    exit_price = Column(Float, nullable=True)  # Price when position was closed
    exit_timestamp = Column(DateTime, nullable=True)  # When position was closed
    profit_loss = Column(Float, nullable=True)  # P&L in quote currency
    profit_loss_percent = Column(Float, nullable=True)  # P&L as percentage

    __table_args__ = (Index("ix_trades_query", "exchange", "symbol", "position_id", "timestamp"),)
