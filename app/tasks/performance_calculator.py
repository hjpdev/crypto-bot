"""
Performance calculation task for the crypto trading bot.

This module provides functionality for calculating and storing performance metrics
on a scheduled basis, including profit and loss, win rate, drawdown, and other KPIs.
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from app.core.exceptions import DatabaseError
from app.core.scheduler import TaskScheduler


class PerformanceCalculator:
    """
    Calculates and stores performance metrics for the trading system.

    Features:
    - Periodic calculation of trading performance metrics
    - Support for different time frames (daily, weekly, monthly, all-time)
    - Comprehensive reporting capabilities
    - Integration with the TaskScheduler for scheduled execution
    """

    def __init__(
        self,
        portfolio_service,
        trade_history_service,
        storage_service,
        config_service,
        calculation_interval_hours: int = 6,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the PerformanceCalculator.

        Args:
            portfolio_service: Service for accessing portfolio data
            trade_history_service: Service for accessing trade history
            storage_service: Service for storing performance metrics
            config_service: Service for accessing system configuration
            calculation_interval_hours: How often to calculate metrics (in hours)
            logger: Logger instance for logging events
        """
        self._portfolio_service = portfolio_service
        self._trade_history_service = trade_history_service
        self._storage_service = storage_service
        self._config_service = config_service
        self._calculation_interval_hours = calculation_interval_hours
        self._logger = logger or logging.getLogger(__name__)

        # State tracking
        self._last_calculation_time: Optional[datetime] = None
        self._last_generated_report_time: Optional[datetime] = None

        self._logger.info(
            f"PerformanceCalculator initialized with interval of {calculation_interval_hours} hours"
        )

    def register_with_scheduler(self, scheduler: TaskScheduler) -> str:
        """
        Register calculation task with the scheduler.

        Args:
            scheduler: TaskScheduler instance

        Returns:
            Task name that can be used to reference the scheduled task
        """
        task_name = scheduler.add_task(
            task_func=self.run,
            interval=self._calculation_interval_hours * 60 * 60,  # Convert to seconds
            name="performance_calculation",
            priority=30,  # Medium priority, lower than market data collection
        )

        # Also register a monthly report generation task
        report_task_name = scheduler.add_task(
            task_func=lambda: self.generate_report(report_type="monthly"),
            interval=24 * 60 * 60,  # Once per day (checks date internally)
            name="monthly_performance_report",
            priority=50,
        )

        self._logger.info(
            f"Registered performance calculation task: {task_name} and "
            f"monthly report task: {report_task_name}"
        )

        return task_name

    def run(self) -> bool:
        """
        Main entry point for scheduled execution.

        Calculates performance metrics and stores them. This method
        is called by the scheduler at regular intervals.

        Returns:
            True if calculation was successful, False otherwise
        """
        self._logger.info("Starting performance metrics calculation")
        start_time = time.time()

        success = self.calculate_metrics()

        duration = time.time() - start_time
        self._logger.info(
            f"Completed performance calculation in {duration:.2f}s. "
            f"Status: {'Success' if success else 'Failure'}"
        )

        if success:
            self._last_calculation_time = datetime.now()

        return success

    def calculate_metrics(self) -> bool:
        """
        Calculate all performance metrics and store them.

        Returns:
            True if calculation was successful, False otherwise
        """
        try:
            # Get required data
            portfolio = self._portfolio_service.get_current_portfolio()
            trades = self._trade_history_service.get_trades(since=self._last_calculation_time)

            # No new trades since last calculation
            if not trades and self._last_calculation_time:
                self._logger.info("No new trades since last calculation, skipping")
                return True

            # Calculate metrics for different time periods
            time_periods = ["daily", "weekly", "monthly", "all_time"]

            for period in time_periods:
                metrics = self._calculate_metrics_for_period(period, portfolio, trades)

                if metrics:
                    # Store the metrics
                    self._storage_service.store_performance_metrics(metrics, period=period)
                    self._logger.debug(f"Stored {period} performance metrics")

            return True

        except DatabaseError as e:
            self._logger.error(f"Database error during performance calculation: {e}")
            return False

        except Exception as e:
            self._logger.exception(f"Unexpected error during performance calculation: {e}")
            return False

    def generate_report(self, report_type: str = "daily") -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Args:
            report_type: Type of report (daily, weekly, monthly, all_time)

        Returns:
            Dictionary containing the performance report data
        """
        try:
            # Only generate monthly report at the beginning of a new month
            if report_type == "monthly":
                today = datetime.now()
                # Skip if it's not the 1st day of the month or if we've already generated today
                if today.day != 1 or (
                    self._last_generated_report_time
                    and self._last_generated_report_time.date() == today.date()
                ):
                    self._logger.debug(
                        f"Skipping monthly report generation (today: {today.strftime('%Y-%m-%d')})"
                    )
                    return {}

            self._logger.info(f"Generating {report_type} performance report")

            # Get metrics for the specified time period
            metrics = self._storage_service.get_performance_metrics(period=report_type)

            if not metrics:
                self._logger.warning(f"No metrics available for {report_type} report")
                return {}

            # Calculate additional metrics for the report
            report = self._build_report(metrics, report_type)

            # Store the report
            self._storage_service.store_performance_report(report, report_type)

            # Mark report as generated
            self._last_generated_report_time = datetime.now()

            self._logger.info(f"Generated {report_type} performance report")

            return report

        except Exception as e:
            self._logger.exception(f"Error generating {report_type} report: {e}")
            return {}

    def _calculate_metrics_for_period(
        self, period: str, portfolio: Dict[str, Any], trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for a specific time period.

        Args:
            period: Time period (daily, weekly, monthly, all_time)
            portfolio: Current portfolio data
            trades: List of trades for analysis

        Returns:
            Dictionary of calculated metrics
        """
        # Filter trades based on period
        filtered_trades = self._filter_trades_by_period(trades, period)

        # If no trades and not all_time, return empty metrics
        if not filtered_trades and period != "all_time":
            return {}

        # Get starting and ending balances
        start_balance, end_balance = self._get_period_balances(period, portfolio)

        # Calculate basic metrics
        total_trades = len(filtered_trades)
        winning_trades = sum(1 for t in filtered_trades if self._is_winning_trade(t))
        losing_trades = total_trades - winning_trades

        # Handle zero division cases
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate profit/loss
        profit_loss = end_balance - start_balance
        profit_loss_pct = (profit_loss / start_balance * 100) if start_balance > 0 else 0

        # Calculate drawdown
        max_drawdown, max_drawdown_pct = self._calculate_drawdown(filtered_trades, start_balance)

        # Average metrics
        avg_profit_winning = self._calculate_average_profit(filtered_trades, only_winning=True)
        avg_loss_losing = self._calculate_average_loss(filtered_trades, only_losing=True)

        # Risk/reward and expectancy
        risk_reward_ratio = abs(avg_profit_winning / avg_loss_losing) if avg_loss_losing != 0 else 0
        expectancy = (win_rate / 100 * avg_profit_winning) + (
            (1 - win_rate / 100) * avg_loss_losing
        )

        # Create metrics object
        metrics = {
            "period": period,
            "timestamp": datetime.now(),
            "start_balance": float(start_balance),
            "end_balance": float(end_balance),
            "profit_loss": float(profit_loss),
            "profit_loss_pct": float(profit_loss_pct),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown_pct),
            "avg_profit_winning": float(avg_profit_winning),
            "avg_loss_losing": float(avg_loss_losing),
            "risk_reward_ratio": float(risk_reward_ratio),
            "expectancy": float(expectancy),
        }

        return metrics

    def _filter_trades_by_period(
        self, trades: List[Dict[str, Any]], period: str
    ) -> List[Dict[str, Any]]:
        """
        Filter trades to only include those in the specified time period.

        Args:
            trades: List of all trades
            period: Time period to filter for

        Returns:
            Filtered list of trades
        """
        now = datetime.now()

        if period == "all_time":
            return trades

        if period == "daily":
            start_time = datetime(now.year, now.month, now.day)
        elif period == "weekly":
            # Start from the most recent Monday
            start_time = now - timedelta(days=now.weekday())
            start_time = datetime(start_time.year, start_time.month, start_time.day)
        elif period == "monthly":
            start_time = datetime(now.year, now.month, 1)
        else:
            raise ValueError(f"Invalid time period: {period}")

        return [
            trade for trade in trades if datetime.fromisoformat(trade["timestamp"]) >= start_time
        ]

    def _get_period_balances(
        self, period: str, current_portfolio: Dict[str, Any]
    ) -> Tuple[Decimal, Decimal]:
        """
        Get starting and ending balances for the specified period.

        Args:
            period: Time period (daily, weekly, monthly, all_time)
            current_portfolio: Current portfolio data

        Returns:
            Tuple of (starting_balance, ending_balance)
        """
        # Current balance is always the ending balance
        end_balance = Decimal(str(current_portfolio.get("total_value", 0)))

        # For all_time, get the initial deposit as starting balance
        if period == "all_time":
            start_balance = Decimal(
                str(self._config_service.get_config().get("initial_deposit", 0))
            )
        else:
            # For other periods, get the balance at period start from history
            historical_balance = self._portfolio_service.get_historical_balance(period)
            start_balance = Decimal(str(historical_balance or end_balance))

        return start_balance, end_balance

    def _is_winning_trade(self, trade: Dict[str, Any]) -> bool:
        """
        Determine if a trade is winning or losing.

        Args:
            trade: Trade data

        Returns:
            True if it's a winning trade, False otherwise
        """
        # Implementation depends on trade data structure
        return trade.get("profit_loss", 0) > 0

    def _calculate_drawdown(
        self, trades: List[Dict[str, Any]], start_balance: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate maximum drawdown over the period.

        Args:
            trades: List of trades
            start_balance: Starting balance

        Returns:
            Tuple of (max_drawdown_value, max_drawdown_percentage)
        """
        if not trades:
            return Decimal("0"), Decimal("0")

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: datetime.fromisoformat(x["timestamp"]))

        peak = start_balance
        max_drawdown = Decimal("0")
        current_balance = start_balance

        # Calculate running balance and track drawdown
        for trade in sorted_trades:
            profit_loss = Decimal(str(trade.get("profit_loss", 0)))
            current_balance += profit_loss

            if current_balance > peak:
                peak = current_balance

            drawdown = peak - current_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate drawdown percentage
        max_drawdown_pct = (max_drawdown / peak * 100) if peak > 0 else Decimal("0")

        return max_drawdown, max_drawdown_pct

    def _calculate_average_profit(
        self, trades: List[Dict[str, Any]], only_winning: bool = False
    ) -> Decimal:
        """
        Calculate average profit per trade.

        Args:
            trades: List of trades
            only_winning: Whether to only consider winning trades

        Returns:
            Average profit as Decimal
        """
        if not trades:
            return Decimal("0")

        if only_winning:
            profitable_trades = [t for t in trades if Decimal(str(t.get("profit_loss", 0))) > 0]
            if not profitable_trades:
                return Decimal("0")

            total_profit = sum(Decimal(str(t.get("profit_loss", 0))) for t in profitable_trades)
            return total_profit / len(profitable_trades)

        # All trades
        total_profit_loss = sum(Decimal(str(t.get("profit_loss", 0))) for t in trades)
        return total_profit_loss / len(trades)

    def _calculate_average_loss(
        self, trades: List[Dict[str, Any]], only_losing: bool = False
    ) -> Decimal:
        """
        Calculate average loss per trade.

        Args:
            trades: List of trades
            only_losing: Whether to only consider losing trades

        Returns:
            Average loss as Decimal (typically negative)
        """
        if not trades:
            return Decimal("0")

        if only_losing:
            losing_trades = [t for t in trades if Decimal(str(t.get("profit_loss", 0))) < 0]
            if not losing_trades:
                return Decimal("0")

            total_loss = sum(Decimal(str(t.get("profit_loss", 0))) for t in losing_trades)
            return total_loss / len(losing_trades)

        # All trades, focusing on losses
        losses = [
            Decimal(str(t.get("profit_loss", 0)))
            for t in trades
            if Decimal(str(t.get("profit_loss", 0))) < 0
        ]

        if not losses:
            return Decimal("0")

        return sum(losses) / len(losses)

    def _build_report(self, metrics: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """
        Build a comprehensive report from metrics data.

        Args:
            metrics: Raw metrics data
            report_type: Type of report

        Returns:
            Dictionary containing the formatted report
        """
        return {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "period_start": self._get_period_start_date(report_type).isoformat(),
            "period_end": datetime.now().isoformat(),
            "summary": {
                "profit_loss": f"{metrics['profit_loss']:.2f}",
                "profit_loss_pct": f"{metrics['profit_loss_pct']:.2f}%",
                "total_trades": metrics["total_trades"],
                "win_rate": f"{metrics['win_rate']:.2f}%",
                "risk_reward_ratio": f"{metrics['risk_reward_ratio']:.2f}",
            },
            "detailed_metrics": metrics,
            "recommendations": self._generate_recommendations(metrics),
        }

    def _get_period_start_date(self, period: str) -> datetime:
        """
        Get the start date for a report period.

        Args:
            period: Period type (daily, weekly, monthly, all_time)

        Returns:
            Start date as datetime
        """
        now = datetime.now()

        if period == "daily":
            return datetime(now.year, now.month, now.day)
        elif period == "weekly":
            # Start from the most recent Monday
            return now - timedelta(days=now.weekday())
        elif period == "monthly":
            return datetime(now.year, now.month, 1)
        elif period == "all_time":
            # Get bot start date from config, or fallback to 1 year ago
            try:
                start_date_str = self._config_service.get_config().get("start_date")
                if start_date_str:
                    return datetime.fromisoformat(start_date_str)
            except (ValueError, TypeError):
                pass

            # Fallback to 1 year ago
            return now - timedelta(days=365)

        # Default to today
        return datetime(now.year, now.month, now.day)

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on performance metrics.

        Args:
            metrics: Performance metrics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Win rate recommendations
        if metrics["win_rate"] < 40:
            recommendations.append(
                "Low win rate detected. Consider reviewing entry criteria and risk management."
            )

        # Risk/reward recommendations
        if metrics["risk_reward_ratio"] < 1:
            recommendations.append(
                "Risk/reward ratio is poor. Consider increasing profit targets or reducing stop loss distances."
            )

        # Drawdown recommendations
        if metrics["max_drawdown_pct"] > 20:
            recommendations.append(
                f"High maximum drawdown of {metrics['max_drawdown_pct']:.2f}%. "
                "Review position sizing and consider reducing risk per trade."
            )

        # Profit/loss recommendations
        if metrics["profit_loss_pct"] < 0:
            recommendations.append(
                "Strategy is currently unprofitable. Consider pausing trading to review performance."
            )

        # Add general recommendation if none specific
        if not recommendations:
            if metrics["profit_loss_pct"] > 0:
                recommendations.append(
                    "Strategy is performing well. Continue monitoring key metrics."
                )
            else:
                recommendations.append(
                    "Performance is neutral. Monitor closely for changes in market conditions."
                )

        return recommendations
