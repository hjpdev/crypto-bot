"""
Position Management Service for the crypto trading bot.

This module provides functionality for managing trading positions,
including opening, updating, and closing positions, as well as
implementing risk management rules and tracking performance.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any

from app.config.config import Config
from app.core.database import Database
from app.models.position import Position, PositionType, PositionStatus, PartialExit
from app.services.risk_manager import RiskManager


class PositionManager:
    """
    Position Manager for handling trading positions.

    This class implements functionality for:
    - Opening and closing positions
    - Tracking position status and performance
    - Implementing risk management rules
    - Managing stop losses and take profits
    - Applying trailing stops

    Attributes:
        config: Configuration instance containing position parameters
        risk_manager: Risk manager for position sizing and risk calculations
        database: Database instance for storing position data
        logger: Logger instance
    """

    def __init__(
        self,
        config: Config,
        risk_manager: RiskManager,
        database: Database,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the PositionManager.

        Args:
            config: Configuration instance containing position parameters
            risk_manager: Risk manager for position sizing and risk calculations
            database: Database instance for storing position data
            logger: Logger instance (if None, a new logger will be created)
        """
        self.config = config
        self.risk_manager = risk_manager
        self.database = database
        self.logger = logger or logging.getLogger("service.position_manager")

        # Load position management configuration
        self._position_config = config.get_nested("position_management", {})

        # Cache for active positions to reduce database queries
        self._active_positions_cache: Dict[int, Position] = {}
        self._cache_valid = False

    def open_position(
        self,
        symbol: str,
        entry_price: Decimal,
        size: Optional[Decimal] = None,
        position_type: PositionType = PositionType.LONG,
        strategy: str = "manual",
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        notes: Optional[str] = None,
        cryptocurrency_id: Optional[int] = None,
    ) -> Position:
        """
        Open a new trading position.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            size: Position size (if None, will be calculated based on risk parameters)
            position_type: LONG or SHORT position
            strategy: Strategy that triggered this position
            stop_loss_price: Stop loss price (if None, will be calculated)
            take_profit_price: Take profit price (if None, will be calculated)
            notes: Additional notes about the position
            cryptocurrency_id: ID of the cryptocurrency (if None, will be looked up)

        Returns:
            The newly created Position object

        Raises:
            ValueError: If position parameters are invalid
        """
        self.logger.info(f"Opening {position_type.value} position for {symbol} at {entry_price}")

        with self.database.get_session() as session:
            # If cryptocurrency_id is not provided, try to look it up
            if cryptocurrency_id is None:
                crypto = session.query("Cryptocurrency").filter_by(symbol=symbol).first()
                if crypto:
                    cryptocurrency_id = crypto.id
                else:
                    self.logger.warning(
                        f"Cryptocurrency not found for {symbol}, using placeholder ID"
                    )
                    cryptocurrency_id = 0  # Placeholder, should be handled better in production

            # Calculate position size if not provided
            if size is None:
                # Calculate stop loss if not provided
                if stop_loss_price is None:
                    stop_loss_price = self._calculate_stop_loss(symbol, entry_price, position_type)

                size = self.risk_manager.calculate_position_size(
                    symbol, entry_price, stop_loss_price
                )
                self.logger.info(f"Calculated position size: {size} based on risk parameters")

            # Calculate take profit if not provided
            if take_profit_price is None:
                take_profit_price = self._calculate_take_profit(symbol, entry_price, position_type)

            # Create the position
            position = Position(
                cryptocurrency_id=cryptocurrency_id,
                symbol=symbol,
                entry_timestamp=datetime.utcnow(),
                entry_price=entry_price,
                size=size,
                position_type=position_type,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                status=PositionStatus.OPEN,
                strategy_used=strategy,
                notes=notes,
            )

            session.add(position)
            session.commit()

            # Invalidate cache
            self._cache_valid = False

            self.logger.info(f"Position opened with ID {position.id}")
            return position

    def update_position(
        self, position_id: int, current_price: Decimal
    ) -> Tuple[Position, Decimal, Decimal]:
        """
        Update a position with the current market price.

        Args:
            position_id: ID of the position to update
            current_price: Current market price

        Returns:
            Tuple of (position, profit_loss, profit_loss_percentage)

        Raises:
            ValueError: If position not found
        """
        with self.database.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()

            if not position:
                raise ValueError(f"Position with ID {position_id} not found")

            if position.status == PositionStatus.CLOSED:
                return position, position.profit_loss, position.profit_loss_percentage

            # Calculate current P/L
            profit_loss, profit_loss_percentage = position.calculate_current_pl(current_price)

            # Check if trailing stop should be adjusted
            if self._should_adjust_trailing_stop(position, current_price):
                self.adjust_trailing_stop(position, current_price)
                session.commit()

            return position, profit_loss, profit_loss_percentage

    def close_position(
        self, position_id: int, exit_price: Decimal, reason: str = "manual"
    ) -> Position:
        """
        Close a position completely.

        Args:
            position_id: ID of the position to close
            exit_price: Exit price for the position
            reason: Reason for closing the position

        Returns:
            The closed Position object

        Raises:
            ValueError: If position not found or already closed
        """
        self.logger.info(f"Closing position {position_id} at price {exit_price}, reason: {reason}")

        with self.database.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()

            if not position:
                raise ValueError(f"Position with ID {position_id} not found")

            if position.status == PositionStatus.CLOSED:
                raise ValueError(f"Position with ID {position_id} is already closed")

            # Apply full exit
            position.apply_exit(price=exit_price, timestamp=datetime.utcnow(), full_exit=True)

            # Update notes with closing reason
            if position.notes:
                position.notes += f"\nClosed: {reason}"
            else:
                position.notes = f"Closed: {reason}"

            session.commit()

            # Invalidate cache
            self._cache_valid = False

            self.logger.info(
                f"Position {position_id} closed with P/L: {position.profit_loss} "
                f"({position.profit_loss_percentage}%)"
            )

            return position

    def apply_partial_exit(
        self, position_id: int, exit_price: Decimal, percentage: Decimal
    ) -> Tuple[Position, PartialExit]:
        """
        Apply a partial exit to a position.

        Args:
            position_id: ID of the position
            exit_price: Exit price for the partial exit
            percentage: Percentage of the position to exit (1-99)

        Returns:
            Tuple of (position, partial_exit)

        Raises:
            ValueError: If position not found, already closed, or invalid percentage
        """
        self.logger.info(
            f"Applying {percentage}% partial exit to position {position_id} at price {exit_price}"
        )

        with self.database.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()

            if not position:
                raise ValueError(f"Position with ID {position_id} not found")

            # Apply partial exit
            partial_exit = position.apply_exit(
                price=exit_price,
                timestamp=datetime.utcnow(),
                full_exit=False,
                exit_percentage=percentage,
            )

            if partial_exit:
                session.add(partial_exit)

            session.commit()

            # Invalidate cache
            self._cache_valid = False

            self.logger.info(
                f"Applied {percentage}% partial exit to position {position_id} "
                f"with P/L: {partial_exit.profit_loss if partial_exit else 'N/A'}"
            )

            return position, partial_exit

    def check_stop_loss(self, position: Position, current_price: Decimal) -> bool:
        """
        Check if stop loss is triggered for a position.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if stop loss is triggered, False otherwise
        """
        if position.status == PositionStatus.CLOSED:
            return False

        if position.position_type == PositionType.LONG:
            return current_price <= position.stop_loss_price
        else:  # SHORT
            return current_price >= position.stop_loss_price

    def check_take_profit(self, position: Position, current_price: Decimal) -> bool:
        """
        Check if take profit is triggered for a position.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if take profit is triggered, False otherwise
        """
        if position.status == PositionStatus.CLOSED:
            return False

        if position.position_type == PositionType.LONG:
            return current_price >= position.take_profit_price
        else:  # SHORT
            return current_price <= position.take_profit_price

    def adjust_trailing_stop(self, position: Position, current_price: Decimal) -> bool:
        """
        Adjust trailing stop for a position if conditions are met.

        Args:
            position: Position to adjust
            current_price: Current market price

        Returns:
            True if trailing stop was adjusted, False otherwise
        """
        # Check if trailing stop is enabled in config
        trailing_stop_enabled = self._position_config.get("trailing_stop_enabled", False)
        if not trailing_stop_enabled:
            return False

        # Get trailing stop parameters
        activation_percentage = Decimal(
            str(self._position_config.get("trailing_stop_activation_percentage", 50))
        )
        trailing_percentage = Decimal(str(self._position_config.get("trailing_stop_percentage", 2)))

        # Calculate profit percentage
        if position.position_type == PositionType.LONG:
            profit_percentage = ((current_price / position.entry_price) - 1) * 100

            # Check if profit exceeds activation threshold
            if profit_percentage >= activation_percentage:
                # Calculate new stop loss level
                new_stop_loss = current_price * (1 - (trailing_percentage / 100))

                # Only update if new stop loss is higher than current stop loss
                if new_stop_loss > position.stop_loss_price:
                    old_stop_loss = position.stop_loss_price
                    position.stop_loss_price = new_stop_loss

                    self.logger.info(
                        f"Adjusted trailing stop for position {position.id} from "
                        f"{old_stop_loss} to {new_stop_loss}"
                    )
                    return True
        else:  # SHORT
            profit_percentage = ((position.entry_price / current_price) - 1) * 100

            # Check if profit exceeds activation threshold
            if profit_percentage >= activation_percentage:
                # Calculate new stop loss level
                new_stop_loss = current_price * (1 + (trailing_percentage / 100))

                # Only update if new stop loss is lower than current stop loss
                if new_stop_loss < position.stop_loss_price:
                    old_stop_loss = position.stop_loss_price
                    position.stop_loss_price = new_stop_loss

                    self.logger.info(
                        f"Adjusted trailing stop for position {position.id} from "
                        f"{old_stop_loss} to {new_stop_loss}"
                    )
                    return True

        return False

    def get_active_positions(self) -> List[Position]:
        """
        Get all currently active positions.

        Returns:
            List of active Position objects
        """
        # Use cache if valid
        if self._cache_valid and self._active_positions_cache:
            return list(self._active_positions_cache.values())

        with self.database.get_session() as session:
            positions = Position.get_open_positions(session)

            # Update cache
            self._active_positions_cache = {p.id: p for p in positions}
            self._cache_valid = True

            return positions

    def get_position_performance(self, position_id: int) -> Dict[str, Any]:
        """
        Get detailed performance metrics for a position.

        Args:
            position_id: ID of the position

        Returns:
            Dictionary with performance metrics

        Raises:
            ValueError: If position not found
        """
        with self.database.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()

            if not position:
                raise ValueError(f"Position with ID {position_id} not found")

            # Calculate holding period
            start_time = position.entry_timestamp
            end_time = position.exit_timestamp or datetime.utcnow()
            holding_period_hours = (end_time - start_time).total_seconds() / 3600

            # Get partial exits
            partial_exits = position.partial_exits

            # Calculate metrics
            performance = {
                "position_id": position.id,
                "symbol": position.symbol,
                "position_type": position.position_type.value,
                "status": position.status.value,
                "entry_price": float(position.entry_price),
                "entry_timestamp": position.entry_timestamp.isoformat(),
                "size": float(position.size),
                "stop_loss_price": float(position.stop_loss_price),
                "take_profit_price": float(position.take_profit_price),
                "holding_period_hours": holding_period_hours,
                "strategy_used": position.strategy_used,
            }

            # Add exit information if position is closed
            if position.status == PositionStatus.CLOSED:
                performance.update(
                    {
                        "exit_price": float(position.exit_price),
                        "exit_timestamp": position.exit_timestamp.isoformat(),
                        "profit_loss": float(position.profit_loss),
                        "profit_loss_percentage": float(position.profit_loss_percentage),
                        "annualized_return": (
                            float(position.profit_loss_percentage) * (8760 / holding_period_hours)
                            if holding_period_hours > 0
                            else 0
                        ),
                    }
                )

            # Add partial exit information
            if partial_exits:
                performance["partial_exits"] = [
                    {
                        "exit_timestamp": pe.exit_timestamp.isoformat(),
                        "exit_price": float(pe.exit_price),
                        "exit_percentage": float(pe.exit_percentage),
                        "profit_loss": float(pe.profit_loss),
                        "trailing_stop_activated": pe.trailing_stop_activated,
                    }
                    for pe in partial_exits
                ]

                # Calculate total realized P/L from partial exits
                total_realized_pl = sum(float(pe.profit_loss) for pe in partial_exits)
                performance["realized_profit_loss"] = total_realized_pl

            return performance

    def _calculate_stop_loss(
        self, symbol: str, entry_price: Decimal, position_type: PositionType
    ) -> Decimal:
        """
        Calculate stop loss price based on configuration and position type.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            position_type: LONG or SHORT position

        Returns:
            Calculated stop loss price
        """
        # Get stop loss percentage from config (default to 5%)
        stop_loss_pct = Decimal(str(self._position_config.get("default_stop_loss_percentage", 5)))

        if position_type == PositionType.LONG:
            return entry_price * (1 - (stop_loss_pct / 100))
        else:  # SHORT
            return entry_price * (1 + (stop_loss_pct / 100))

    def _calculate_take_profit(
        self, symbol: str, entry_price: Decimal, position_type: PositionType
    ) -> Decimal:
        """
        Calculate take profit price based on configuration and position type.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            position_type: LONG or SHORT position

        Returns:
            Calculated take profit price
        """
        # Get take profit percentage from config (default to 10%)
        take_profit_pct = Decimal(
            str(self._position_config.get("default_take_profit_percentage", 10))
        )

        if position_type == PositionType.LONG:
            return entry_price * (1 + (take_profit_pct / 100))
        else:  # SHORT
            return entry_price * (1 - (take_profit_pct / 100))

    def _should_adjust_trailing_stop(self, position: Position, current_price: Decimal) -> bool:
        """
        Determine if trailing stop should be adjusted based on current price.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if trailing stop should be adjusted, False otherwise
        """
        # Check if trailing stop is enabled in config
        trailing_stop_enabled = self._position_config.get("trailing_stop_enabled", False)
        if not trailing_stop_enabled:
            return False

        # Get trailing stop activation percentage
        activation_percentage = Decimal(
            str(self._position_config.get("trailing_stop_activation_percentage", 50))
        )

        # Calculate current profit percentage
        if position.position_type == PositionType.LONG:
            profit_percentage = ((current_price / position.entry_price) - 1) * 100
            return profit_percentage >= activation_percentage
        else:  # SHORT
            profit_percentage = ((position.entry_price / current_price) - 1) * 100
            return profit_percentage >= activation_percentage
