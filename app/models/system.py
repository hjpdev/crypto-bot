from datetime import datetime
from typing import Dict, List, Optional, Any, TypeVar, Type, Union
from sqlalchemy import Column, DateTime, String, Text, Integer, Numeric, JSON, desc
from sqlalchemy.ext.mutable import MutableDict

from app.models.base_model import BaseModel
from app.core.database import get_db

T = TypeVar("T", bound="ConfigurationHistory")
P = TypeVar("P", bound="PerformanceMetrics")


class ConfigurationHistory(BaseModel):
    """
    Model for storing historical configuration snapshots.

    Attributes:
        timestamp (DateTime): When the configuration was saved
        configuration (JSON): The complete configuration object
        run_id (String): Unique identifier for each bot run
        notes (Text): Optional notes about this configuration
    """

    __tablename__ = "configuration_history"

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    configuration = Column(MutableDict.as_mutable(JSON), nullable=False)
    run_id = Column(String(64), nullable=False, index=True)
    notes = Column(Text, nullable=True)

    @classmethod
    def save_current_config(
        cls: Type[T], config: Dict[str, Any], run_id: str, notes: Optional[str] = None
    ) -> T:
        """
        Save a snapshot of the current configuration.

        Args:
            config: The complete configuration dictionary
            run_id: Unique identifier for the current bot run
            notes: Optional notes about this configuration

        Returns:
            ConfigurationHistory: The saved configuration history object
        """
        session = next(get_db())
        try:
            # Make a deep copy to ensure we don't have references to the original dict
            config_copy = {}
            if config:
                import copy

                config_copy = copy.deepcopy(config)

            config_history = cls(configuration=config_copy, run_id=run_id, notes=notes)
            session.add(config_history)
            session.commit()
            session.refresh(config_history)
            return config_history
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    @classmethod
    def get_latest(cls: Type[T]) -> Optional[T]:
        """Get the most recent configuration."""
        session = next(get_db())
        try:
            return session.query(cls).order_by(desc(cls.timestamp)).first()
        finally:
            session.close()

    @classmethod
    def get_by_run_id(cls: Type[T], run_id: str) -> List[T]:
        """
        Get all configurations for a specific run.

        Args:
            run_id: Unique identifier for the bot run

        Returns:
            List[ConfigurationHistory]: List of configurations for the run
        """
        session = next(get_db())
        try:
            return session.query(cls).filter(cls.run_id == run_id).order_by(cls.timestamp).all()
        finally:
            session.close()


class PerformanceMetrics(BaseModel):
    """
    Model for storing trading performance metrics.

    Attributes:
        timestamp (DateTime): When the metrics were recorded
        run_id (String): Unique identifier matching ConfigurationHistory
        total_trades (Integer): Total number of trades
        winning_trades (Integer): Number of winning trades
        losing_trades (Integer): Number of losing trades
        win_rate (Numeric): Percentage of winning trades
        average_profit (Numeric): Average profit per winning trade
        average_loss (Numeric): Average loss per losing trade
        profit_factor (Numeric): Ratio of gross profit to gross loss
        max_drawdown (Numeric): Maximum drawdown percentage
        sharpe_ratio (Numeric): Risk-adjusted return metric
        total_profit_loss (Numeric): Total profit/loss
    """

    __tablename__ = "performance_metrics"

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Numeric(10, 2), nullable=False, default=0)
    average_profit = Column(Numeric(20, 8), nullable=False, default=0)
    average_loss = Column(Numeric(20, 8), nullable=False, default=0)
    profit_factor = Column(Numeric(10, 2), nullable=False, default=0)
    max_drawdown = Column(Numeric(10, 2), nullable=False, default=0)
    sharpe_ratio = Column(Numeric(10, 4), nullable=False, default=0)
    total_profit_loss = Column(Numeric(20, 8), nullable=False, default=0)

    @classmethod
    def record_current_performance(
        cls: Type[P], metrics_dict: Dict[str, Union[int, float]], run_id: str
    ) -> P:
        """
        Save current performance metrics.

        Args:
            metrics_dict: Dictionary containing performance metrics
            run_id: Unique identifier for the current bot run

        Returns:
            PerformanceMetrics: The saved performance metrics object

        Raises:
            ValueError: If required metrics are missing or invalid
        """
        session = next(get_db())
        try:
            # Validate required metrics
            required_fields = [
                "total_trades",
                "winning_trades",
                "losing_trades",
                "win_rate",
                "total_profit_loss",
            ]

            for field in required_fields:
                if field not in metrics_dict:
                    raise ValueError(f"Required metric '{field}' missing from metrics_dict")

            # Create metrics object
            metrics = cls(
                run_id=run_id,
                total_trades=metrics_dict.get("total_trades", 0),
                winning_trades=metrics_dict.get("winning_trades", 0),
                losing_trades=metrics_dict.get("losing_trades", 0),
                win_rate=metrics_dict.get("win_rate", 0),
                average_profit=metrics_dict.get("average_profit", 0),
                average_loss=metrics_dict.get("average_loss", 0),
                profit_factor=metrics_dict.get("profit_factor", 0),
                max_drawdown=metrics_dict.get("max_drawdown", 0),
                sharpe_ratio=metrics_dict.get("sharpe_ratio", 0),
                total_profit_loss=metrics_dict.get("total_profit_loss", 0),
            )

            session.add(metrics)
            session.commit()
            session.refresh(metrics)
            return metrics
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    @classmethod
    def get_latest(cls: Type[P]) -> Optional[P]:
        """
        Get the most recent performance metrics.

        Returns:
            Optional[PerformanceMetrics]: The most recent metrics or None
        """
        session = next(get_db())
        try:
            return session.query(cls).order_by(desc(cls.timestamp)).first()
        finally:
            session.close()

    @classmethod
    def get_by_run_id(cls: Type[P], run_id: str) -> List[P]:
        """
        Get all performance metrics for a specific run.

        Args:
            run_id: Unique identifier for the bot run

        Returns:
            List[PerformanceMetrics]: List of metrics for the run
        """
        session = next(get_db())
        try:
            return session.query(cls).filter(cls.run_id == run_id).order_by(cls.timestamp).all()
        finally:
            session.close()

    @staticmethod
    def calculate_from_trades(trades: List[Dict[str, Any]]) -> Dict[str, Union[int, float]]:
        """
        Calculate performance metrics from a list of trade dictionaries.

        Args:
            trades: List of trade dictionaries with at least 'profit_loss' field

        Returns:
            Dict: Dictionary with calculated performance metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "average_profit": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "total_profit_loss": 0,
            }

        # Extract profit/loss values
        pl_values = [trade.get("profit_loss", 0) for trade in trades]

        # Basic counts
        total_trades = len(trades)
        winning_trades = sum(1 for pl in pl_values if pl > 0)
        losing_trades = sum(1 for pl in pl_values if pl < 0)

        # Win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Average profit/loss
        winning_values = [pl for pl in pl_values if pl > 0]
        losing_values = [pl for pl in pl_values if pl < 0]

        average_profit = sum(winning_values) / len(winning_values) if winning_values else 0
        average_loss = sum(losing_values) / len(losing_values) if losing_values else 0

        # Profit factor
        gross_profit = sum(winning_values)
        gross_loss = abs(sum(losing_values)) if losing_values else 0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss != 0
            else 0 if gross_profit == 0 else float("inf")
        )

        # Total P/L
        total_profit_loss = sum(pl_values)

        # Maximum drawdown calculation (simplified approach)
        cumulative = [0]
        for pl in pl_values:
            cumulative.append(cumulative[-1] + pl)

        max_drawdown = 0
        peak = cumulative[0]

        for value in cumulative[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

        # Simplified Sharpe ratio calculation (assumes daily returns)
        # In a real system, this would need proper return calculation and risk-free rate
        returns = [p2 - p1 for p1, p2 in zip(cumulative, cumulative[1:])]
        mean_return = sum(returns) / len(returns) if returns else 0
        std_dev = (
            (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0
        )
        sharpe_ratio = (mean_return / std_dev) if std_dev != 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_profit_loss": total_profit_loss,
        }
