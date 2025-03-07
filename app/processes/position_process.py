"""
Position management process implementation for the crypto trading bot.

This module implements a process that periodically updates and manages
active trading positions using the PositionManager service.
"""

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Any

from app.config.config import Config
from app.core.database import Database
from app.core.exceptions import ExchangeError
from app.core.process import BaseProcess
from app.models.position import PositionStatus
from app.services.exchange_service import ExchangeService
from app.services.position_manager import PositionManager
from app.services.risk_manager import RiskManager


class PositionProcess(BaseProcess):
    """
    Process that periodically manages active trading positions.

    This process runs at configured intervals to:
    - Update all active positions with current market prices
    - Check for exit conditions (stop loss, take profit)
    - Adjust trailing stops if needed
    - Record position changes to the database
    - Handle errors gracefully and recover from failures
    """

    def __init__(
        self,
        config: Config,
        exchange_service: ExchangeService,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        database: Database,
        interval_seconds: float = 60.0,  # Default to 1 minute
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the position management process.

        Args:
            config: Application configuration
            exchange_service: Service for interacting with exchanges
            position_manager: Service for managing positions
            risk_manager: Service for risk management
            database: Database instance for storing position data
            interval_seconds: Time between process iterations in seconds
            logger: Logger instance
        """
        super().__init__(
            name="position_manager",
            interval_seconds=interval_seconds,
            logger=logger or logging.getLogger("process.position"),
        )

        self.config = config
        self.exchange_service = exchange_service
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.database = database

        # Process statistics
        self._total_positions_processed = 0
        self._total_positions_closed = 0
        self._total_trailing_stops_adjusted = 0
        self._last_run_positions_count = 0
        self._last_run_duration = 0.0

        # Cache for current prices to reduce API calls
        self._price_cache: Dict[str, Dict[str, Any]] = {}
        self._price_cache_timestamp = datetime.utcnow()
        self._price_cache_ttl = 10  # Cache TTL in seconds

        self.logger.info(f"Position process initialized with interval of {interval_seconds}s")

    def _run_iteration(self) -> None:
        """
        Run a single iteration of the position management process.

        This method is called by the BaseProcess at the configured interval.
        """
        start_time = time.time()
        self.logger.info("Starting position management iteration")

        try:
            # Process active positions
            positions_count = self.process_active_positions()

            # Check for exit conditions
            exits_applied = self.check_exit_conditions()

            # Record position changes
            self.record_position_changes()

            # Update statistics
            self._total_positions_processed += positions_count
            self._last_run_positions_count = positions_count

            # Log summary
            self.logger.info(
                f"Position iteration completed: {positions_count} positions processed, "
                f"{exits_applied} exits applied"
            )

        except Exception as e:
            self.logger.error(f"Error in position management process: {str(e)}", exc_info=True)
            # Don't raise the exception - let the BaseProcess handle error recovery

        finally:
            # Calculate and store duration
            self._last_run_duration = time.time() - start_time
            self.logger.debug(f"Position iteration took {self._last_run_duration:.2f} seconds")

    def process_active_positions(self) -> int:
        """
        Process all active positions.

        This method:
        1. Retrieves all active positions
        2. Gets current market prices
        3. Updates position status and P/L
        4. Adjusts trailing stops if needed

        Returns:
            Number of positions processed
        """
        # Get all active positions
        active_positions = self.position_manager.get_active_positions()

        if not active_positions:
            self.logger.info("No active positions to process")
            return 0

        self.logger.info(f"Processing {len(active_positions)} active positions")

        # Process each position
        for position in active_positions:
            try:
                # Get current price for the symbol
                current_price = self._get_current_price(position.symbol)

                if current_price is None:
                    self.logger.warning(
                        f"Could not get current price for {position.symbol}, skipping position {position.id}"
                    )
                    continue

                # Update position with current price
                position, profit_loss, profit_loss_percentage = (
                    self.position_manager.update_position(position.id, current_price)
                )

                self.logger.debug(
                    f"Position {position.id} ({position.symbol}) updated: "
                    f"P/L: {profit_loss} ({profit_loss_percentage:.2f}%)"
                )

            except Exception as e:
                self.logger.error(
                    f"Error processing position {position.id} ({position.symbol}): {str(e)}",
                    exc_info=True,
                )

        return len(active_positions)

    def check_exit_conditions(self) -> int:
        """
        Check exit conditions for all active positions.

        This method:
        1. Checks if stop loss or take profit levels are reached
        2. Applies exits as needed
        3. Records exit reasons

        Returns:
            Number of exits applied
        """
        # Get all active positions
        active_positions = self.position_manager.get_active_positions()

        if not active_positions:
            return 0

        exits_applied = 0

        # Check each position for exit conditions
        for position in active_positions:
            try:
                # Get current price for the symbol
                current_price = self._get_current_price(position.symbol)

                if current_price is None:
                    continue

                # Check stop loss
                if self.position_manager.check_stop_loss(position, current_price):
                    self.logger.info(
                        f"Stop loss triggered for position {position.id} ({position.symbol}) "
                        f"at price {current_price}"
                    )

                    # Close position
                    self.position_manager.close_position(
                        position.id, current_price, reason="stop_loss"
                    )

                    exits_applied += 1
                    self._total_positions_closed += 1
                    continue  # Skip further checks for this position

                # Check take profit
                if self.position_manager.check_take_profit(position, current_price):
                    self.logger.info(
                        f"Take profit triggered for position {position.id} ({position.symbol}) "
                        f"at price {current_price}"
                    )

                    # Check if we should apply partial take profit
                    partial_tp_enabled = self.config.get_nested(
                        "position_management.partial_take_profit_enabled", False
                    )

                    if partial_tp_enabled and position.status == PositionStatus.OPEN:
                        # Get partial take profit percentage
                        partial_tp_percentage = Decimal(
                            str(
                                self.config.get_nested(
                                    "position_management.partial_take_profit_percentage", 50
                                )
                            )
                        )

                        self.logger.info(
                            f"Applying {partial_tp_percentage}% partial take profit for "
                            f"position {position.id}"
                        )

                        # Apply partial exit
                        self.position_manager.apply_partial_exit(
                            position.id, current_price, partial_tp_percentage
                        )
                    else:
                        # Close position completely
                        self.position_manager.close_position(
                            position.id, current_price, reason="take_profit"
                        )
                        self._total_positions_closed += 1

                    exits_applied += 1

            except Exception as e:
                self.logger.error(
                    f"Error checking exit conditions for position {position.id}: {str(e)}",
                    exc_info=True,
                )

        return exits_applied

    def record_position_changes(self) -> None:
        """
        Record position changes to the database.

        This method ensures all position changes are properly persisted
        and can be used for reporting and analysis.
        """
        # Most position changes are already recorded in the database
        # through the PositionManager methods. This method can be used
        # for additional recording needs, such as:

        # 1. Record position snapshots for historical analysis
        # 2. Generate position change events for other components
        # 3. Update aggregate statistics

        try:
            # Get active positions for statistics
            active_positions = self.position_manager.get_active_positions()
            active_count = len(active_positions)

            # Record basic statistics
            with self.database.get_session() as session:  # noqa: F841
                # Example: Update system statistics table
                # This is a placeholder - implement according to your schema
                self.logger.debug(f"Recording position statistics: {active_count} active positions")

                # You could record things like:
                # - Total active positions
                # - Total position value
                # - Aggregate profit/loss
                # - Position distribution by symbol

        except Exception as e:
            self.logger.error(f"Error recording position changes: {str(e)}", exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about the process.

        Returns:
            Dictionary with process status and statistics
        """
        # Get base status from parent class
        status = super().get_status()

        # Add position-specific status information
        status.update(
            {
                "total_positions_processed": self._total_positions_processed,
                "total_positions_closed": self._total_positions_closed,
                "total_trailing_stops_adjusted": self._total_trailing_stops_adjusted,
                "last_run_positions_count": self._last_run_positions_count,
                "last_run_duration_seconds": self._last_run_duration,
                "price_cache_size": len(self._price_cache),
            }
        )

        return status

    def _on_start(self) -> None:
        """
        Called when the process starts.

        Performs initialization tasks before the main loop starts.
        """
        self.logger.info("Position management process starting")

        # Clear caches
        self._price_cache.clear()

        # Reset statistics
        self._last_run_positions_count = 0
        self._last_run_duration = 0.0

    def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get the current price for a symbol, using cache when possible.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price or None if not available
        """
        # Check if we need to refresh the cache
        current_time = datetime.utcnow()
        cache_age = (current_time - self._price_cache_timestamp).total_seconds()

        if cache_age > self._price_cache_ttl:
            # Cache expired, clear it
            self._price_cache.clear()
            self._price_cache_timestamp = current_time

        # Check if price is in cache
        if symbol in self._price_cache:
            return Decimal(str(self._price_cache[symbol]["price"]))

        # Fetch price from exchange
        try:
            ticker = self.exchange_service.fetch_ticker(symbol)

            if ticker and "last" in ticker:
                price = Decimal(str(ticker["last"]))

                # Update cache
                self._price_cache[symbol] = {
                    "price": float(price),
                    "timestamp": current_time,
                }

                return price

            return None

        except ExchangeError as e:
            self.logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return None
