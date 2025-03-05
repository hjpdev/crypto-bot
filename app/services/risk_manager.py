"""
Risk Management Service for the crypto trading bot.

This module provides risk management functionality, including position sizing,
stop loss calculations, take profit levels, and risk limit enforcement.
"""

from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal
import pandas as pd

from app.models.position import Position, PositionType
from app.config.config import Config


class RiskManager:
    """
    Risk Manager for handling position sizing and risk-related calculations.

    This class implements various risk management techniques including:
    - Position sizing based on risk parameters
    - Stop loss calculations (fixed, volatility-based, indicator-based)
    - Take profit calculations with partial exit levels
    - Trailing stop logic
    - Position limits enforcement

    Attributes:
        config: Configuration instance containing risk parameters
    """

    def __init__(self, config: Config):
        """
        Initialize the RiskManager.

        Args:
            config: Configuration instance containing risk parameters
        """
        self.config = config
        self._risk_config = config.get_nested("risk_management", {})

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        risk_per_trade: Optional[Decimal] = None,
        account_balance: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price level
            risk_per_trade: Risk percentage per trade (overrides config if provided)
            account_balance: Account balance to use (overrides config if provided)

        Returns:
            Position size in base currency

        Raises:
            ValueError: If stop loss price is not valid or calculation fails
        """
        # Use provided values or fall back to config
        risk_pct = risk_per_trade or Decimal(str(self._risk_config.get("risk_per_trade", 1.0)))
        balance = account_balance or Decimal(str(self._risk_config.get("account_balance", 0.0)))

        if balance <= 0:
            raise ValueError("Account balance must be greater than zero")

        # Calculate risk amount in quote currency
        risk_amount = balance * (risk_pct / Decimal("100"))

        # Calculate price difference between entry and stop loss
        if entry_price <= 0 or stop_loss_price <= 0:
            raise ValueError("Entry price and stop loss price must be greater than zero")

        price_diff = abs(entry_price - stop_loss_price)

        if price_diff == 0:
            raise ValueError("Entry price cannot be equal to stop loss price")

        # Calculate position size in base currency
        risk_per_unit = price_diff / entry_price  # Risk per unit as a percentage of entry price
        position_size = risk_amount / (entry_price * risk_per_unit)

        # Apply minimum and maximum position size limits if configured
        min_size = Decimal(
            str(
                self._risk_config.get(
                    f"min_position_size.{symbol}",
                    self._risk_config.get("min_position_size.default", 0),
                )
            )
        )
        max_size = Decimal(
            str(
                self._risk_config.get(
                    f"max_position_size.{symbol}",
                    self._risk_config.get("max_position_size.default", float("inf")),
                )
            )
        )

        position_size = max(min_size, min(position_size, max_size))

        return position_size

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: Decimal,
        direction: PositionType,
        market_data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> Decimal:
        """
        Calculate the stop loss level based on configured strategy.

        Supports multiple stop loss types:
        - Fixed percentage
        - ATR-based (volatility)
        - Support/Resistance levels
        - Moving average based

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price of the position
            direction: Position type (LONG or SHORT)
            market_data: DataFrame with recent market data including indicators
            config: Optional custom configuration for stop loss calculation

        Returns:
            Stop loss price
        """
        stop_config = config or self._risk_config.get("stop_loss", {})
        stop_type = stop_config.get("type", "fixed_percentage")

        if stop_type == "fixed_percentage":
            percentage = Decimal(str(stop_config.get("percentage", 2.0)))
            if direction == PositionType.LONG:
                return entry_price * (Decimal("1") - percentage / Decimal("100"))
            else:
                return entry_price * (Decimal("1") + percentage / Decimal("100"))

        elif stop_type == "atr_based":
            # Get ATR from market data
            atr_column = stop_config.get("atr_column", "atr")
            multiplier = Decimal(str(stop_config.get("multiplier", 2.0)))

            if atr_column not in market_data.columns:
                raise ValueError(f"ATR column '{atr_column}' not found in market data")

            atr_value = Decimal(str(market_data[atr_column].iloc[-1]))

            if direction == PositionType.LONG:
                return entry_price - (atr_value * multiplier)
            else:
                return entry_price + (atr_value * multiplier)

        elif stop_type == "indicator_based":
            indicator_column = stop_config.get("indicator_column")
            if not indicator_column or indicator_column not in market_data.columns:
                raise ValueError(f"Indicator column '{indicator_column}' not found in market data")

            indicator_value = Decimal(str(market_data[indicator_column].iloc[-1]))
            return indicator_value

        elif stop_type == "support_resistance":
            # Find the nearest support/resistance level
            levels_column = stop_config.get("levels_column", "support_resistance_levels")

            if levels_column not in market_data.columns:
                raise ValueError(
                    f"Support/resistance column '{levels_column}' not found in market data"
                )

            levels = market_data[levels_column].iloc[-1]

            if isinstance(levels, list):
                # Find appropriate support/resistance level based on direction
                if direction == PositionType.LONG:
                    # Find the highest support below entry price
                    supports = [level for level in levels if Decimal(str(level)) < entry_price]
                    if supports:
                        return Decimal(str(max(supports)))
                else:
                    # Find the lowest resistance above entry price
                    resistances = [level for level in levels if Decimal(str(level)) > entry_price]
                    if resistances:
                        return Decimal(str(min(resistances)))

            # Fallback to fixed percentage if no suitable level found
            percentage = Decimal(str(stop_config.get("fallback_percentage", 2.0)))
            if direction == PositionType.LONG:
                return entry_price * (Decimal("1") - percentage / Decimal("100"))
            else:
                return entry_price * (Decimal("1") + percentage / Decimal("100"))

        else:
            raise ValueError(f"Unsupported stop loss type: {stop_type}")

    def calculate_take_profit_levels(
        self,
        entry_price: Decimal,
        direction: PositionType,
        stop_loss: Decimal,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Decimal, Decimal]]:
        """
        Calculate take profit levels with partial exit points.

        Args:
            entry_price: Entry price of the position
            direction: Position type (LONG or SHORT)
            stop_loss: Stop loss price level
            config: Optional custom configuration for take profit calculation

        Returns:
            List of tuples containing (price_level, exit_percentage)
        """
        tp_config = config or self._risk_config.get("take_profit", {})

        # Check if using risk-reward based take profits
        if tp_config.get("type", "risk_reward") == "risk_reward":
            risk = abs(entry_price - stop_loss)
            levels = []

            # Generate take profit levels based on risk:reward ratios
            for level in tp_config.get(
                "levels", [{"ratio": 2.0, "percentage": 50}, {"ratio": 3.0, "percentage": 50}]
            ):
                ratio = Decimal(str(level.get("ratio", 1.0)))
                percentage = Decimal(str(level.get("percentage", 100.0)))

                if direction == PositionType.LONG:
                    price = entry_price + (risk * ratio)
                else:
                    price = entry_price - (risk * ratio)

                levels.append((price, percentage))

            return levels

        # Fixed percentage based take profits
        elif tp_config.get("type") == "fixed_percentage":
            levels = []

            for level in tp_config.get(
                "levels",
                [
                    {"percentage": 3.0, "exit_percentage": 50},
                    {"percentage": 5.0, "exit_percentage": 50},
                ],
            ):
                price_pct = Decimal(str(level.get("percentage", 3.0)))
                exit_pct = Decimal(str(level.get("exit_percentage", 50.0)))

                if direction == PositionType.LONG:
                    price = entry_price * (Decimal("1") + price_pct / Decimal("100"))
                else:
                    price = entry_price * (Decimal("1") - price_pct / Decimal("100"))

                levels.append((price, exit_pct))

            return levels

        # Default to a single take profit at 2x the risk
        else:
            risk = abs(entry_price - stop_loss)
            if direction == PositionType.LONG:
                price = entry_price + (risk * Decimal("2"))
            else:
                price = entry_price - (risk * Decimal("2"))

            return [(price, Decimal("100"))]

    def adjust_trailing_stop(
        self, position: Position, current_price: Decimal, config: Optional[Dict[str, Any]] = None
    ) -> Optional[Decimal]:
        """
        Adjust the trailing stop if conditions are met.

        Args:
            position: Current position
            current_price: Current market price
            config: Optional custom configuration for trailing stop

        Returns:
            New stop loss price or None if no adjustment needed
        """
        trailing_config = config or self._risk_config.get("trailing_stop", {})

        if not trailing_config.get("enabled", False):
            return None

        activation_percentage = Decimal(str(trailing_config.get("activation_percentage", 1.0)))
        trailing_percentage = Decimal(str(trailing_config.get("trailing_percentage", 0.5)))

        # Calculate how far price has moved in our favor
        if position.position_type == PositionType.LONG:
            price_movement = (
                (current_price - position.entry_price) / position.entry_price * Decimal("100")
            )

            # Check if price has moved enough to activate trailing stop
            if price_movement >= activation_percentage:
                # Calculate new stop loss based on trailing percentage
                new_stop = current_price * (Decimal("1") - trailing_percentage / Decimal("100"))

                # Only move stop loss up, never down
                if new_stop > position.stop_loss_price:
                    return new_stop

        else:  # SHORT position
            price_movement = (
                (position.entry_price - current_price) / position.entry_price * Decimal("100")
            )

            if price_movement >= activation_percentage:
                new_stop = current_price * (Decimal("1") + trailing_percentage / Decimal("100"))

                # Only move stop loss down, never up
                if new_stop < position.stop_loss_price:
                    return new_stop

        return None

    def validate_trade(
        self, trade_params: Dict[str, Any], portfolio_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a potential trade against risk limits.

        Args:
            trade_params: Dictionary with trade parameters
            portfolio_state: Current portfolio state

        Returns:
            Tuple of (is_valid, error_message)
        """
        symbol = trade_params.get("symbol")
        size = trade_params.get("size", Decimal("0"))
        direction = trade_params.get("position_type")

        # Check if we have necessary inputs
        if not symbol or not size or not direction:
            return False, "Missing required trade parameters"

        # Check maximum open positions limit
        max_positions = self._risk_config.get("max_open_positions", 0)
        if max_positions > 0 and portfolio_state.get("open_positions_count", 0) >= max_positions:
            return False, f"Maximum open positions limit reached ({max_positions})"

        # Check symbol exposure limit
        max_symbol_exposure = self._risk_config.get(
            f"max_exposure.{symbol}", self._risk_config.get("max_exposure.default", 0)
        )

        if max_symbol_exposure > 0:
            current_exposure = portfolio_state.get("exposure_per_symbol", {}).get(
                symbol, Decimal("0")
            )
            max_symbol_value = Decimal(str(max_symbol_exposure))

            if current_exposure + size > max_symbol_value:
                return False, f"Maximum exposure for {symbol} would be exceeded"

        # Check for position in opposite direction
        if self._risk_config.get("prevent_opposite_positions", True):
            opposite_exists = False
            for pos in portfolio_state.get("open_positions", []):
                if pos.get("symbol") == symbol and pos.get("position_type") != direction:
                    opposite_exists = True
                    break

            if opposite_exists:
                return False, f"Position in opposite direction already exists for {symbol}"

        # Check total portfolio exposure
        max_exposure = self._risk_config.get("max_total_exposure", 0)
        if max_exposure > 0:
            current_total = portfolio_state.get("total_exposure", Decimal("0"))
            if current_total + size > Decimal(str(max_exposure)):
                return False, "Maximum total portfolio exposure would be exceeded"

        return True, None

    def calculate_portfolio_exposure(self, positions: List[Position]) -> Dict[str, Any]:
        """
        Calculate current portfolio exposure metrics.

        Args:
            positions: List of open positions

        Returns:
            Dictionary with exposure metrics
        """
        result = {
            "total_exposure": Decimal("0"),
            "exposure_per_symbol": {},
            "open_positions_count": len(positions),
            "exposure_ratio": Decimal("0"),  # Exposure as percentage of configured balance
        }

        # Calculate exposures
        for position in positions:
            symbol = position.symbol
            size = position.size

            # Update total exposure
            result["total_exposure"] += size

            # Update per-symbol exposure
            if symbol not in result["exposure_per_symbol"]:
                result["exposure_per_symbol"][symbol] = Decimal("0")

            result["exposure_per_symbol"][symbol] += size

        # Calculate exposure ratio if balance is configured
        account_balance = Decimal(str(self._risk_config.get("account_balance", 0)))
        if account_balance > 0:
            result["exposure_ratio"] = (result["total_exposure"] / account_balance) * Decimal("100")

        return result

    def should_adjust_position_size(
        self, current_volatility: Decimal, baseline_volatility: Decimal
    ) -> Decimal:
        """
        Determine if position size should be adjusted based on market volatility.

        Higher volatility = smaller position sizes
        Lower volatility = larger position sizes

        Args:
            current_volatility: Current market volatility (e.g. ATR value)
            baseline_volatility: Baseline volatility for comparison

        Returns:
            Adjustment multiplier (1.0 = no adjustment)
        """
        if baseline_volatility <= 0:
            return Decimal("1")

        volatility_ratio = current_volatility / baseline_volatility

        # If feature not enabled, return no adjustment
        if not self._risk_config.get("volatility_adjustment", {}).get("enabled", False):
            return Decimal("1")

        max_adjustment = Decimal(
            str(self._risk_config.get("volatility_adjustment", {}).get("max_adjustment", 2.0))
        )
        min_adjustment = Decimal(
            str(self._risk_config.get("volatility_adjustment", {}).get("min_adjustment", 0.5))
        )

        # Calculate a non-linear adjustment - exponential decrease for higher volatility
        if volatility_ratio > Decimal("1"):
            # Higher volatility than baseline - reduce position size
            multiplier = Decimal("1") / (volatility_ratio ** Decimal("0.5"))
            return max(min_adjustment, multiplier)
        else:
            # Lower volatility than baseline - increase position size
            multiplier = Decimal("1") / (volatility_ratio ** Decimal("0.5"))
            return min(max_adjustment, multiplier)
