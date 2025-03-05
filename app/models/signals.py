from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from decimal import Decimal


class SignalType(str, Enum):
    """Enum for signal types."""

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    EXIT = "EXIT"
    PARTIAL_EXIT = "PARTIAL_EXIT"


class Signal(BaseModel):
    """
    Represents a trading signal.

    A trading signal indicates a potential trading opportunity and includes
    information about the type, strength, and source of the signal.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "json_encoders": {datetime: lambda v: v.isoformat(), Decimal: lambda v: float(v)}
        }
    )

    symbol: str
    type: SignalType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    timeframe: str
    price: Decimal
    indicators: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Validate that confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def as_dict(self) -> Dict[str, Any]:
        """Convert the signal to a dictionary."""
        return {
            "symbol": self.symbol,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "source": self.source,
            "timeframe": self.timeframe,
            "price": float(self.price),
            "indicators": self.indicators,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create a Signal from a dictionary."""
        # Convert string signal type to enum if needed
        if isinstance(data.get("type"), str):
            data["type"] = SignalType(data["type"].upper())

        # Convert timestamp string to datetime if needed
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

        # Convert price to Decimal if it's a float or string
        if "price" in data and not isinstance(data["price"], Decimal):
            data["price"] = Decimal(str(data["price"]))

        return cls(**data)

    def is_valid(self) -> bool:
        """Check if the signal is valid."""
        # Signal must be recent (within last hour)
        current_time = datetime.utcnow()
        if (current_time - self.timestamp).total_seconds() > 3600:  # 1 hour in seconds
            return False

        # Signal must have sufficient confidence
        if self.confidence < 0.5:
            return False

        return True

    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal."""
        return self.type == SignalType.BUY

    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal."""
        return self.type == SignalType.SELL

    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal."""
        return self.type in [SignalType.EXIT, SignalType.PARTIAL_EXIT]


class SignalCollection:
    """
    A collection of trading signals for analysis.

    This class provides functionality to store, filter, and analyze multiple trading signals.
    """

    def __init__(self, signals: Optional[List[Signal]] = None):
        """Initialize the signal collection."""
        self.signals = signals or []

    def add_signal(self, signal: Signal) -> None:
        """Add a signal to the collection."""
        self.signals.append(signal)

    def get_signals_by_type(self, signal_type: SignalType) -> List[Signal]:
        """Get signals of a specific type."""
        return [signal for signal in self.signals if signal.type == signal_type]

    def get_signals_for_symbol(self, symbol: str) -> List[Signal]:
        """Get signals for a specific symbol."""
        return [signal for signal in self.signals if signal.symbol == symbol]

    def get_recent_signals(self, hours: int = 24) -> List[Signal]:
        """Get signals from the last specified hours."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - datetime.timedelta(hours=hours)

        return [signal for signal in self.signals if signal.timestamp >= cutoff_time]

    def get_strongest_signal(self, symbol: Optional[str] = None) -> Optional[Signal]:
        """Get the strongest signal (highest confidence) for a symbol or overall."""
        filtered_signals = self.signals
        if symbol:
            filtered_signals = self.get_signals_for_symbol(symbol)

        if not filtered_signals:
            return None

        return max(filtered_signals, key=lambda s: s.confidence)

    def get_consensus_type(self, symbol: str) -> SignalType:
        """
        Determine the consensus signal type for a symbol.

        This method analyzes all signals for a given symbol and determines
        the consensus (majority) signal type.
        """
        symbol_signals = self.get_signals_for_symbol(symbol)

        if not symbol_signals:
            return SignalType.NEUTRAL

        # Count signal types
        type_counts = {}
        for signal in symbol_signals:
            signal_type = signal.type
            if signal_type not in type_counts:
                type_counts[signal_type] = 0
            type_counts[signal_type] += signal.confidence

        if not type_counts:
            return SignalType.NEUTRAL

        # Return the type with the highest weighted count
        return max(type_counts.items(), key=lambda x: x[1])[0]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all signals to a list of dictionaries."""
        return [signal.as_dict() for signal in self.signals]

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict[str, Any]]) -> "SignalCollection":
        """Create a signal collection from a list of dictionaries."""
        signals = [Signal.from_dict(signal_dict) for signal_dict in dict_list]
        return cls(signals)
