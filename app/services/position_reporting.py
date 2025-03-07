"""
Position Reporting Service for the crypto trading bot.

This module provides functionality for generating reports and analytics
on trading positions, including performance metrics and aggregate statistics.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional, Any

from sqlalchemy import func, and_, desc

from app.config.config import Config
from app.core.database import Database
from app.models.position import Position, PositionStatus


class PositionReporting:
    """
    Position Reporting for analyzing trading performance.

    This class implements functionality for:
    - Generating position performance reports
    - Calculating aggregate statistics
    - Tracking daily/weekly/monthly P&L

    Attributes:
        config: Configuration instance
        database: Database instance for retrieving position data
        logger: Logger instance
    """

    def __init__(
        self,
        config: Config,
        database: Database,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the PositionReporting service.

        Args:
            config: Configuration instance
            database: Database instance for retrieving position data
            logger: Logger instance (if None, a new logger will be created)
        """
        self.config = config
        self.database = database
        self.logger = logger or logging.getLogger("service.position_reporting")

    def generate_position_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of current positions.

        Returns:
            A dictionary containing summary information about positions:
            - total_positions: Total number of positions
            - open_positions: Number of open positions
            - partially_closed_positions: Number of partially closed positions
            - closed_positions: Number of closed positions
            - total_profit_loss: Total P&L across all positions
            - average_profit_loss_percentage: Average P&L percentage
            - winning_positions: Number of winning positions
            - losing_positions: Number of losing positions
            - win_rate: Percentage of winning positions
        """
        self.logger.info("Generating position summary")

        with self.database.get_session() as session:
            # Get position counts by status
            total_positions = session.query(func.count(Position.id)).scalar() or 0
            open_positions = (
                session.query(func.count(Position.id))
                .filter(Position.status == PositionStatus.OPEN)
                .scalar()
                or 0
            )
            partially_closed_positions = (
                session.query(func.count(Position.id))
                .filter(Position.status == PositionStatus.PARTIALLY_CLOSED)
                .scalar()
                or 0
            )
            closed_positions = (
                session.query(func.count(Position.id))
                .filter(Position.status == PositionStatus.CLOSED)
                .scalar()
                or 0
            )

            # Get total P&L for closed positions
            total_profit_loss = session.query(func.sum(Position.profit_loss)).filter(
                Position.status == PositionStatus.CLOSED
            ).scalar() or Decimal("0")

            # Get average P&L percentage for closed positions
            avg_profit_loss_percentage = session.query(
                func.avg(Position.profit_loss_percentage)
            ).filter(Position.status == PositionStatus.CLOSED).scalar() or Decimal("0")

            # Count winning and losing positions
            winning_positions = (
                session.query(func.count(Position.id))
                .filter(and_(Position.status == PositionStatus.CLOSED, Position.profit_loss > 0))
                .scalar()
                or 0
            )

            losing_positions = (
                session.query(func.count(Position.id))
                .filter(and_(Position.status == PositionStatus.CLOSED, Position.profit_loss <= 0))
                .scalar()
                or 0
            )

            # Calculate win rate
            win_rate = Decimal("0")
            if closed_positions > 0:
                win_rate = Decimal(str(winning_positions)) / Decimal(str(closed_positions)) * 100

        return {
            "total_positions": total_positions,
            "open_positions": open_positions,
            "partially_closed_positions": partially_closed_positions,
            "closed_positions": closed_positions,
            "total_profit_loss": total_profit_loss,
            "average_profit_loss_percentage": avg_profit_loss_percentage,
            "winning_positions": winning_positions,
            "losing_positions": losing_positions,
            "win_rate": win_rate,
        }

    def calculate_daily_pnl(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate profit and loss for a specific day.

        Args:
            date: The date to calculate P&L for (defaults to today)

        Returns:
            A dictionary containing daily P&L information:
            - date: The date
            - closed_positions: Number of positions closed on this date
            - total_profit_loss: Total P&L for the day
            - best_position: Details of the best performing position
            - worst_position: Details of the worst performing position
        """
        if date is None:
            date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        next_day = date + timedelta(days=1)
        self.logger.info(f"Calculating daily P&L for {date.date()}")

        with self.database.get_session() as session:
            # Count positions closed on this day
            closed_positions_count = (
                session.query(func.count(Position.id))
                .filter(
                    and_(
                        Position.exit_timestamp >= date,
                        Position.exit_timestamp < next_day,
                        Position.status == PositionStatus.CLOSED,
                    )
                )
                .scalar()
                or 0
            )

            # Calculate total P&L for the day
            total_profit_loss = session.query(func.sum(Position.profit_loss)).filter(
                and_(
                    Position.exit_timestamp >= date,
                    Position.exit_timestamp < next_day,
                    Position.status == PositionStatus.CLOSED,
                )
            ).scalar() or Decimal("0")

            # Get best performing position for the day
            best_position = (
                session.query(Position)
                .filter(
                    and_(
                        Position.exit_timestamp >= date,
                        Position.exit_timestamp < next_day,
                        Position.status == PositionStatus.CLOSED,
                    )
                )
                .order_by(desc(Position.profit_loss_percentage))
                .first()
            )

            # Get worst performing position for the day
            worst_position = (
                session.query(Position)
                .filter(
                    and_(
                        Position.exit_timestamp >= date,
                        Position.exit_timestamp < next_day,
                        Position.status == PositionStatus.CLOSED,
                    )
                )
                .order_by(Position.profit_loss_percentage)
                .first()
            )

        result = {
            "date": date.date().isoformat(),
            "closed_positions": closed_positions_count,
            "total_profit_loss": total_profit_loss,
            "best_position": None,
            "worst_position": None,
        }

        if best_position:
            result["best_position"] = {
                "id": best_position.id,
                "symbol": best_position.symbol,
                "profit_loss": best_position.profit_loss,
                "profit_loss_percentage": best_position.profit_loss_percentage,
            }

        if worst_position:
            result["worst_position"] = {
                "id": worst_position.id,
                "symbol": worst_position.symbol,
                "profit_loss": worst_position.profit_loss,
                "profit_loss_percentage": worst_position.profit_loss_percentage,
            }

        return result

    def generate_performance_report(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed performance report for a date range.

        Args:
            start_date: Start date for the report (default: 30 days ago)
            end_date: End date for the report (default: today)

        Returns:
            A dictionary containing performance metrics:
            - time_period: Description of the time period
            - total_trades: Number of closed trades in the period
            - winning_trades: Number of winning trades
            - losing_trades: Number of losing trades
            - win_rate: Percentage of winning trades
            - total_profit_loss: Total P&L for the period
            - average_profit_loss: Average P&L per trade
            - profit_factor: Ratio of gross profit to gross loss
            - largest_winner: Details of the largest winning trade
            - largest_loser: Details of the largest losing trade
            - average_holding_time: Average position holding time in hours
            - daily_pnl: List of daily P&L values
            - symbols: Performance breakdown by symbol
            - strategies: Performance breakdown by strategy
        """
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        self.logger.info(
            f"Generating performance report from {start_date.date()} to {end_date.date()}"
        )

        with self.database.get_session() as session:
            # Get all closed positions in the time period
            positions = (
                session.query(Position)
                .filter(
                    and_(
                        Position.exit_timestamp >= start_date,
                        Position.exit_timestamp <= end_date,
                        Position.status == PositionStatus.CLOSED,
                    )
                )
                .all()
            )

            total_trades = len(positions)
            if total_trades == 0:
                return {
                    "time_period": f"{start_date.date()} to {end_date.date()}",
                    "total_trades": 0,
                    "message": "No closed trades in this period",
                }

            # Calculate basic metrics
            total_profit_loss = sum(p.profit_loss for p in positions)
            winning_trades = sum(1 for p in positions if p.profit_loss > 0)
            losing_trades = sum(1 for p in positions if p.profit_loss <= 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

            # Calculate profit factor
            gross_profit = sum(p.profit_loss for p in positions if p.profit_loss > 0)
            gross_loss = abs(sum(p.profit_loss for p in positions if p.profit_loss < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Find largest winner and loser
            largest_winner = max(
                (p for p in positions if p.profit_loss > 0),
                key=lambda p: p.profit_loss,
                default=None,
            )
            largest_loser = min(
                (p for p in positions if p.profit_loss < 0),
                key=lambda p: p.profit_loss,
                default=None,
            )

            # Calculate average holding time
            holding_times = [
                (p.exit_timestamp - p.entry_timestamp).total_seconds() / 3600  # in hours
                for p in positions
            ]
            avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

            # Group performance by symbol
            symbol_performance = {}
            for p in positions:
                if p.symbol not in symbol_performance:
                    symbol_performance[p.symbol] = {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "total_profit_loss": Decimal("0"),
                    }
                symbol_performance[p.symbol]["total_trades"] += 1
                if p.profit_loss > 0:
                    symbol_performance[p.symbol]["winning_trades"] += 1
                symbol_performance[p.symbol]["total_profit_loss"] += p.profit_loss

            # Group performance by strategy
            strategy_performance = {}
            for p in positions:
                if p.strategy_used not in strategy_performance:
                    strategy_performance[p.strategy_used] = {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "total_profit_loss": Decimal("0"),
                    }
                strategy_performance[p.strategy_used]["total_trades"] += 1
                if p.profit_loss > 0:
                    strategy_performance[p.strategy_used]["winning_trades"] += 1
                strategy_performance[p.strategy_used]["total_profit_loss"] += p.profit_loss

        # Calculate daily P&L
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_day = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_pnl = []

        while current_date <= end_date_day:
            daily_result = self.calculate_daily_pnl(current_date)
            daily_pnl.append(
                {
                    "date": daily_result["date"],
                    "profit_loss": daily_result["total_profit_loss"],
                    "closed_positions": daily_result["closed_positions"],
                }
            )
            current_date += timedelta(days=1)

        report = {
            "time_period": f"{start_date.date()} to {end_date.date()}",
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit_loss": total_profit_loss,
            "average_profit_loss": total_profit_loss / total_trades if total_trades > 0 else 0,
            "profit_factor": profit_factor,
            "average_holding_time": avg_holding_time,
            "daily_pnl": daily_pnl,
            "symbols": symbol_performance,
            "strategies": strategy_performance,
        }

        if largest_winner:
            report["largest_winner"] = {
                "id": largest_winner.id,
                "symbol": largest_winner.symbol,
                "profit_loss": largest_winner.profit_loss,
                "profit_loss_percentage": largest_winner.profit_loss_percentage,
                "strategy": largest_winner.strategy_used,
            }

        if largest_loser:
            report["largest_loser"] = {
                "id": largest_loser.id,
                "symbol": largest_loser.symbol,
                "profit_loss": largest_loser.profit_loss,
                "profit_loss_percentage": largest_loser.profit_loss_percentage,
                "strategy": largest_loser.strategy_used,
            }

        return report
