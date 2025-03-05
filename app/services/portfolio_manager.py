"""
Portfolio Management Service for the crypto trading bot.

This module provides functionality for tracking and managing the trading portfolio,
including position tracking, exposure limits, and diversification rules.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from decimal import Decimal
from datetime import datetime
import uuid

from app.models.position import Position, PositionStatus, PositionType
from app.config.config import Config


class PortfolioManager:
    """
    Portfolio Manager for tracking and managing the trading portfolio.

    This class implements portfolio management functionality including:
    - Position tracking
    - Exposure limits enforcement
    - Diversification rules
    - Position updates and lifecycle management

    Attributes:
        config: Configuration instance containing portfolio parameters
        positions: Dictionary of tracked positions by ID
    """

    def __init__(self, config: Config):
        """
        Initialize the PortfolioManager.

        Args:
            config: Configuration instance containing portfolio parameters
        """
        self.config = config
        self._portfolio_config = config.get_nested("portfolio_management", {})
        self.positions: Dict[str, Position] = {}

    def add_position(self, position: Position) -> str:
        """
        Add a new position to the portfolio.

        Args:
            position: Position object to add

        Returns:
            Position ID

        Raises:
            ValueError: If position doesn't meet portfolio requirements
        """
        # Verify position has required attributes
        if not position.symbol or not position.size or not position.position_type:
            raise ValueError("Position missing required attributes")

        # Check if position meets portfolio limits
        can_add, reason = self.check_position_limits(position)
        if not can_add:
            raise ValueError(f"Position exceeds portfolio limits: {reason}")

        # Generate an ID if not provided
        if not position.id:
            position.id = str(uuid.uuid4())

        # Add to tracked positions
        self.positions[position.id] = position

        return position.id

    def update_position(self, position_id: str, updates: Dict[str, Any]) -> Position:
        """
        Update an existing position.

        Args:
            position_id: ID of the position to update
            updates: Dictionary of field updates

        Returns:
            Updated position

        Raises:
            KeyError: If position_id doesn't exist
        """
        if position_id not in self.positions:
            raise KeyError(f"Position with ID {position_id} not found")

        position = self.positions[position_id]

        # Apply updates
        for field, value in updates.items():
            if hasattr(position, field):
                setattr(position, field, value)

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: Decimal,
        exit_timestamp: Optional[datetime] = None,
        partial_exit: bool = False,
        exit_percentage: Optional[Decimal] = None,
    ) -> Position:
        """
        Close a position entirely or partially.

        Args:
            position_id: ID of the position to close
            exit_price: Exit price of the position
            exit_timestamp: Timestamp of the exit (defaults to now)
            partial_exit: Whether this is a partial exit
            exit_percentage: Percentage to exit (required if partial_exit is True)

        Returns:
            Updated position

        Raises:
            KeyError: If position_id doesn't exist
            ValueError: If partial exit parameters are invalid
        """
        if position_id not in self.positions:
            raise KeyError(f"Position with ID {position_id} not found")

        position = self.positions[position_id]
        timestamp = exit_timestamp or datetime.utcnow()

        if partial_exit:
            if not exit_percentage or exit_percentage <= 0 or exit_percentage > 100:
                raise ValueError("Valid exit_percentage required for partial exit")

            # Apply partial exit to the position
            partial_exit = position.apply_exit(
                price=exit_price,
                timestamp=timestamp,
                full_exit=False,
                exit_percentage=exit_percentage,
            )

            # If position is now fully closed, remove from active positions
            if position.status == PositionStatus.CLOSED:
                del self.positions[position_id]

        else:
            # Apply full exit
            position.apply_exit(price=exit_price, timestamp=timestamp, full_exit=True)

            # Remove closed position from active tracking
            del self.positions[position_id]

        return position

    def get_current_exposure(self) -> Dict[str, Any]:
        """
        Get the current portfolio exposure metrics.

        Returns:
            Dictionary with exposure metrics:
            - total_exposure: Total position size
            - exposure_per_symbol: Dictionary of exposure per symbol
            - symbols_count: Number of unique symbols
            - positions_count: Number of open positions
            - exposure_ratio: Exposure as percentage of account balance
        """
        result = {
            "total_exposure": Decimal("0"),
            "exposure_per_symbol": {},
            "symbols_count": 0,
            "positions_count": len(self.positions),
            "exposure_ratio": Decimal("0"),
        }

        symbols: Set[str] = set()

        # Calculate exposures
        for position in self.positions.values():
            symbol = position.symbol
            symbols.add(symbol)
            size = position.size

            # Update total exposure
            result["total_exposure"] += size

            # Update per-symbol exposure
            if symbol not in result["exposure_per_symbol"]:
                result["exposure_per_symbol"][symbol] = Decimal("0")

            result["exposure_per_symbol"][symbol] += size

        result["symbols_count"] = len(symbols)

        # Calculate exposure ratio if balance is configured
        account_balance = Decimal(str(self._portfolio_config.get("account_balance", 0)))
        if account_balance > 0:
            result["exposure_ratio"] = (result["total_exposure"] / account_balance) * Decimal("100")

        return result

    def get_exposure_per_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get exposure metrics for a specific symbol.

        Args:
            symbol: Symbol to get exposure for

        Returns:
            Dictionary with symbol exposure metrics:
            - total_exposure: Total position size for the symbol
            - long_exposure: Long position size
            - short_exposure: Short position size
            - net_exposure: Net exposure (long - short)
            - positions_count: Number of open positions for the symbol
            - ratio_of_portfolio: Percentage of total portfolio allocated to the symbol
        """
        result = {
            "total_exposure": Decimal("0"),
            "long_exposure": Decimal("0"),
            "short_exposure": Decimal("0"),
            "net_exposure": Decimal("0"),
            "positions_count": 0,
            "ratio_of_portfolio": Decimal("0"),
        }

        # Find positions for this symbol
        for position in self.positions.values():
            if position.symbol == symbol:
                result["positions_count"] += 1
                size = position.size

                # Track total exposure
                result["total_exposure"] += size

                # Track directional exposure
                if position.position_type == PositionType.LONG:
                    result["long_exposure"] += size
                else:
                    result["short_exposure"] += size

        # Calculate net exposure
        result["net_exposure"] = result["long_exposure"] - result["short_exposure"]

        # Calculate ratio of portfolio if we have exposure
        portfolio_exposure = self.get_current_exposure()["total_exposure"]
        if portfolio_exposure > 0:
            result["ratio_of_portfolio"] = (
                result["total_exposure"] / portfolio_exposure
            ) * Decimal("100")

        return result

    def check_position_limits(self, new_position: Position) -> Tuple[bool, Optional[str]]:
        """
        Check if a new position would exceed portfolio limits.

        Args:
            new_position: Position to check

        Returns:
            Tuple of (is_within_limits, reason)
        """
        symbol = new_position.symbol
        position_type = new_position.position_type
        size = new_position.size

        # Get current exposure metrics
        current_exposure = self.get_current_exposure()
        symbol_exposure = self.get_exposure_per_symbol(symbol)

        # Check maximum positions limit
        max_positions = self._portfolio_config.get("max_positions", 0)
        if max_positions > 0 and current_exposure["positions_count"] >= max_positions:
            return False, f"Maximum positions limit reached ({max_positions})"

        # Check maximum symbols limit
        max_symbols = self._portfolio_config.get("max_symbols", 0)
        if max_symbols > 0:
            # Check if this would add a new symbol
            if (
                symbol not in current_exposure["exposure_per_symbol"]
                and current_exposure["symbols_count"] >= max_symbols
            ):
                return False, f"Maximum symbols limit reached ({max_symbols})"

        # Check symbol exposure limit
        max_symbol_exposure = self._portfolio_config.get(
            f"max_exposure.{symbol}", self._portfolio_config.get("max_exposure.default", 0)
        )

        if max_symbol_exposure > 0:
            if symbol_exposure["total_exposure"] + size > Decimal(str(max_symbol_exposure)):
                return False, f"Maximum exposure for {symbol} would be exceeded"

        # Check for position in opposite direction if not allowed
        if self._portfolio_config.get("prevent_opposite_positions", True):
            has_long = symbol_exposure["long_exposure"] > 0
            has_short = symbol_exposure["short_exposure"] > 0

            if (has_long and position_type == PositionType.SHORT) or (
                has_short and position_type == PositionType.LONG
            ):
                return False, f"Position in opposite direction already exists for {symbol}"

        # Check maximum positions per symbol
        max_positions_per_symbol = self._portfolio_config.get("max_positions_per_symbol", 0)
        if (
            max_positions_per_symbol > 0
            and symbol_exposure["positions_count"] >= max_positions_per_symbol
        ):
            return False, f"Maximum positions for {symbol} reached ({max_positions_per_symbol})"

        # Check total portfolio exposure
        max_exposure = self._portfolio_config.get("max_total_exposure", 0)
        if max_exposure > 0:
            if current_exposure["total_exposure"] + size > Decimal(str(max_exposure)):
                return False, "Maximum total portfolio exposure would be exceeded"

        # Check maximum exposure ratio
        max_exposure_ratio = self._portfolio_config.get("max_exposure_ratio", 0)
        if max_exposure_ratio > 0:
            account_balance = Decimal(str(self._portfolio_config.get("account_balance", 0)))
            if account_balance > 0:
                new_ratio = (
                    (current_exposure["total_exposure"] + size) / account_balance
                ) * Decimal("100")
                if new_ratio > Decimal(str(max_exposure_ratio)):
                    return (
                        False,
                        f"Maximum exposure ratio would be exceeded ({max_exposure_ratio}%)",
                    )

        # Check maximum concentration per symbol
        max_concentration = self._portfolio_config.get("max_symbol_concentration", 0)
        if max_concentration > 0:
            new_symbol_exposure = symbol_exposure["total_exposure"] + size
            new_total_exposure = current_exposure["total_exposure"] + size

            # Skip concentration check if this is the first position
            if new_total_exposure > 0 and current_exposure["positions_count"] > 0:
                concentration = (new_symbol_exposure / new_total_exposure) * Decimal("100")
                if concentration > Decimal(str(max_concentration)):
                    return (
                        False,
                        f"Maximum concentration for {symbol} would be exceeded ({max_concentration}%)",
                    )

        return True, None

    def get_all_positions(self) -> List[Position]:
        """
        Get all currently tracked positions.

        Returns:
            List of Position objects
        """
        return list(self.positions.values())

    def get_position(self, position_id: str) -> Position:
        """
        Get a specific position by ID.

        Args:
            position_id: ID of the position to retrieve

        Returns:
            Position object

        Raises:
            KeyError: If position_id doesn't exist
        """
        if position_id not in self.positions:
            raise KeyError(f"Position with ID {position_id} not found")

        return self.positions[position_id]

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get all positions for a specific symbol.

        Args:
            symbol: Symbol to filter positions by

        Returns:
            List of Position objects
        """
        return [p for p in self.positions.values() if p.symbol == symbol]

    def get_positions_by_type(self, position_type: PositionType) -> List[Position]:
        """
        Get all positions of a specific type.

        Args:
            position_type: Position type to filter by

        Returns:
            List of Position objects
        """
        return [p for p in self.positions.values() if p.position_type == position_type]

    def load_positions(self, positions: List[Position]) -> None:
        """
        Load multiple positions into the portfolio manager.

        This is useful for initializing the portfolio state from the database.

        Args:
            positions: List of Position objects to load

        Notes:
            This bypasses position limit checks, so should only be used for initialization
        """
        for position in positions:
            if position.id:
                self.positions[position.id] = position
            else:
                # Generate an ID if not available
                position.id = str(uuid.uuid4())
                self.positions[position.id] = position
