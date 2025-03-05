from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
import logging
from decimal import Decimal

from app.models.position import Position


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface that all trading strategies must implement.
    Each concrete strategy will provide its own implementation of these methods.

    Attributes:
        name: The name of the strategy
        description: A brief description of the strategy
        config: Configuration parameters for the strategy
        logger: Logger instance for the strategy
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration parameters."""
        self.name = self.__class__.__name__
        self.description = "Base trading strategy"
        self.config = config
        self.logger = logging.getLogger(f"strategy.{self.name}")
        self.validate_config()

    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_params = ["risk_per_trade", "max_open_positions"]

        for param in required_params:
            if param not in self.config:
                raise ValueError(
                    f"Required parameter '{param}' not found in strategy configuration"
                )

        if not (0 < self.config["risk_per_trade"] <= 5):
            raise ValueError("risk_per_trade must be between 0 and 5 percent")

        if not (1 <= self.config["max_open_positions"] <= 20):
            raise ValueError("max_open_positions must be between 1 and 20")

    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate buy/sell signals based on market data.

        Args:
            market_data: Dictionary containing market data for analysis

        Returns:
            A list of signal dictionaries containing signal details
        """
        pass

    @abstractmethod
    def should_enter_position(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if a position should be entered for a given symbol.

        Args:
            symbol: The trading symbol to evaluate
            market_data: Dictionary containing market data for analysis

        Returns:
            A tuple containing:
                - A boolean indicating if a position should be entered
                - A dictionary with additional information about the decision
        """
        pass

    @abstractmethod
    def should_exit_position(
        self, position: Position, market_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if an existing position should be exited.

        Args:
            position: The current position to evaluate
            market_data: Dictionary containing market data for analysis

        Returns:
            A tuple containing:
                - A boolean indicating if the position should be exited
                - A dictionary with additional information about the decision
        """
        pass

    @abstractmethod
    def calculate_position_size(self, symbol: str, account_balance: Decimal) -> Decimal:
        """Calculate the position size for a trade."""
        pass

    @abstractmethod
    def get_stop_loss(
        self, symbol: str, entry_price: Decimal, market_data: Dict[str, Any]
    ) -> Decimal:
        """Determine the stop loss price for a trade."""
        pass

    @abstractmethod
    def get_take_profit(
        self, symbol: str, entry_price: Decimal, market_data: Dict[str, Any]
    ) -> Union[Decimal, List[Decimal]]:
        """
        Determine the take profit level(s) for a trade.

        Args:
            symbol: The trading symbol
            entry_price: The entry price for the trade
            market_data: Dictionary containing market data for analysis

        Returns:
            Either a single take profit price or a list of take profit prices
        """
        pass

    def calculate_risk_amount(self, account_balance: Decimal) -> Decimal:
        """Calculate the amount of capital to risk on a trade."""
        risk_percent = self.config["risk_per_trade"]
        return account_balance * Decimal(risk_percent) / Decimal(100)

    def calculate_risk_reward_ratio(
        self, entry_price: Decimal, stop_loss: Decimal, take_profit: Decimal
    ) -> Decimal:
        """Calculate the risk-to-reward ratio for a trade."""
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            raise ValueError("Prices must be positive")

        # For long positions
        if entry_price > stop_loss and take_profit > entry_price:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        # For short positions
        elif entry_price < stop_loss and take_profit < entry_price:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        else:
            raise ValueError("Invalid price configuration for risk-reward calculation")

        if risk == 0:
            raise ValueError("Risk cannot be zero")

        return reward / risk

    def is_valid_trade_setup(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        min_risk_reward: Decimal = Decimal("1.5"),
    ) -> bool:
        """Validate if a trade setup meets the minimum criteria."""
        try:
            risk_reward = self.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
            return risk_reward >= min_risk_reward
        except ValueError:
            return False

    def get_timeframes(self) -> List[str]:
        """
        Get the timeframes required by this strategy.

        Returns:
            A list of timeframe strings (e.g., ["1m", "5m", "1h"])
        """
        return self.config.get("timeframes", ["1h"])

    def get_required_indicators(self) -> List[str]:
        """
        Get the indicators required by this strategy.

        Returns:
            A list of indicator names (e.g., ["ema_50", "rsi_14", "macd"])
        """
        return self.config.get("indicators", [])

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the strategy configuration."""
        self.config.update(new_config)
        self.validate_config()
        self.logger.info(f"Strategy configuration updated: {self.config}")
