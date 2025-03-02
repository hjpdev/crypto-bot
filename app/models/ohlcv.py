from sqlalchemy import Column, String, Float, DateTime, Index, UniqueConstraint

from app.models.base_model import BaseModel


class OHLCV(BaseModel):
    """
    Model for storing OHLCV (Open, High, Low, Close, Volume) market data.
    """

    __tablename__ = "ohlcv"

    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "timeframe", "timestamp", name="uix_ohlcv"),
        Index("ix_ohlcv_query", "exchange", "symbol", "timeframe", "timestamp"),
    )
