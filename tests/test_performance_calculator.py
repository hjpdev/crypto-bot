"""Tests for the PerformanceCalculator class."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from app.core.exceptions import DatabaseError
from app.core.scheduler import TaskScheduler
from app.tasks.performance_calculator import PerformanceCalculator


class TestPerformanceCalculator:
    """Tests for the PerformanceCalculator class."""

    @pytest.fixture
    def mock_portfolio_service(self):
        """Fixture to provide a mock portfolio service."""
        mock = MagicMock()
        mock.get_current_portfolio.return_value = {
            "total_value": 10500.0,
            "base_currency": "USDT",
            "holdings": {
                "BTC": {"amount": 0.5, "value_in_base": 8000.0},
                "ETH": {"amount": 2.0, "value_in_base": 2000.0},
                "USDT": {"amount": 500.0, "value_in_base": 500.0},
            },
        }
        mock.get_historical_balance.return_value = 10000.0
        return mock

    @pytest.fixture
    def mock_trade_history_service(self):
        """Fixture to provide a mock trade history service."""
        mock = MagicMock()

        # Sample trades covering different scenarios
        sample_trades = [
            {
                "id": "1",
                "symbol": "BTC/USDT",
                "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                "side": "buy",
                "price": 16000.0,
                "amount": 0.25,
                "cost": 4000.0,
                "fee": 4.0,
                "profit_loss": 500.0,  # Profitable trade
            },
            {
                "id": "2",
                "symbol": "ETH/USDT",
                "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                "side": "buy",
                "price": 1000.0,
                "amount": 2.0,
                "cost": 2000.0,
                "fee": 2.0,
                "profit_loss": -200.0,  # Losing trade
            },
            {
                "id": "3",
                "symbol": "BTC/USDT",
                "timestamp": datetime.now().isoformat(),
                "side": "buy",
                "price": 16800.0,
                "amount": 0.25,
                "cost": 4200.0,
                "fee": 4.2,
                "profit_loss": 200.0,  # Recent profitable trade
            },
        ]

        mock.get_trades.return_value = sample_trades
        return mock

    @pytest.fixture
    def mock_storage_service(self):
        """Fixture to provide a mock storage service."""
        mock = MagicMock()

        # Mock the get_performance_metrics method to return sample metrics
        sample_metrics = {
            "period": "daily",
            "timestamp": datetime.now().isoformat(),
            "start_balance": 10000.0,
            "end_balance": 10500.0,
            "profit_loss": 500.0,
            "profit_loss_pct": 5.0,
            "total_trades": 3,
            "winning_trades": 2,
            "losing_trades": 1,
            "win_rate": 66.67,
            "max_drawdown": 200.0,
            "max_drawdown_pct": 2.0,
            "avg_profit_winning": 350.0,
            "avg_loss_losing": -200.0,
            "risk_reward_ratio": 1.75,
            "expectancy": 166.67,
        }

        mock.get_performance_metrics.return_value = sample_metrics
        return mock

    @pytest.fixture
    def mock_config_service(self):
        """Fixture to provide a mock config service."""
        mock = MagicMock()
        mock.get_config.return_value = {
            "initial_deposit": 10000.0,
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        }
        return mock

    @pytest.fixture
    def mock_logger(self):
        """Fixture to provide a mock logger."""
        return MagicMock()

    @pytest.fixture
    def calculator(self, mock_portfolio_service, mock_trade_history_service,
                   mock_storage_service, mock_config_service, mock_logger):
        """Fixture to create a PerformanceCalculator with mocked dependencies."""
        return PerformanceCalculator(
            portfolio_service=mock_portfolio_service,
            trade_history_service=mock_trade_history_service,
            storage_service=mock_storage_service,
            config_service=mock_config_service,
            calculation_interval_hours=6,
            logger=mock_logger,
        )

    def test_init(self, calculator, mock_portfolio_service, mock_trade_history_service,
                  mock_storage_service, mock_config_service, mock_logger):
        """Test initializing the PerformanceCalculator."""
        assert calculator._portfolio_service == mock_portfolio_service
        assert calculator._trade_history_service == mock_trade_history_service
        assert calculator._storage_service == mock_storage_service
        assert calculator._config_service == mock_config_service
        assert calculator._calculation_interval_hours == 6
        assert calculator._logger == mock_logger
        assert calculator._last_calculation_time is None
        assert calculator._last_generated_report_time is None

    def test_register_with_scheduler(self, calculator, mock_logger):
        """Test registering tasks with the scheduler."""
        scheduler = TaskScheduler(logger=mock_logger)

        task_name = calculator.register_with_scheduler(scheduler)

        assert task_name == "performance_calculation"
        assert "performance_calculation" in scheduler._tasks
        assert "monthly_performance_report" in scheduler._tasks
        assert scheduler._tasks["performance_calculation"].interval == 21600  # 6 hours in seconds
        assert scheduler._tasks["monthly_performance_report"].interval == 86400  # 24 hours in seconds
        assert scheduler._tasks["performance_calculation"].priority == 30
        assert scheduler._tasks["monthly_performance_report"].priority == 50

    def test_run(self, calculator, monkeypatch):
        """Test running the performance calculation."""
        # Mock the calculate_metrics method
        mock_calculate = MagicMock(return_value=True)
        monkeypatch.setattr(calculator, "calculate_metrics", mock_calculate)

        # Run the calculation
        result = calculator.run()

        assert result is True
        mock_calculate.assert_called_once()
        assert calculator._last_calculation_time is not None

        # Test with failed calculation
        mock_calculate.reset_mock()
        mock_calculate.return_value = False

        # Reset the last calculation time
        calculator._last_calculation_time = None

        result = calculator.run()

        assert result is False
        mock_calculate.assert_called_once()
        assert calculator._last_calculation_time is None  # Should not be updated on failure

    def test_calculate_metrics(self, calculator, mock_portfolio_service,
                               mock_trade_history_service, mock_storage_service):
        """Test calculating performance metrics."""
        # Mock _calculate_metrics_for_period to return some metrics
        metrics = {
            "period": "daily",
            "timestamp": datetime.now(),
            "start_balance": 10000.0,
            "end_balance": 10500.0,
            "profit_loss": 500.0,
            "profit_loss_pct": 5.0,
            "total_trades": 3,
            "winning_trades": 2,
            "losing_trades": 1,
            "win_rate": 66.67,
            "max_drawdown": 200.0,
            "max_drawdown_pct": 2.0,
            "avg_profit_winning": 350.0,
            "avg_loss_losing": -200.0,
            "risk_reward_ratio": 1.75,
            "expectancy": 166.67,
        }

        calculator._calculate_metrics_for_period = MagicMock(return_value=metrics)

        # Test successful calculation
        result = calculator.calculate_metrics()

        assert result is True
        # Should have fetched portfolio and trades
        mock_portfolio_service.get_current_portfolio.assert_called_once()
        mock_trade_history_service.get_trades.assert_called_once()

        # Should have stored metrics for each time period
        assert mock_storage_service.store_performance_metrics.call_count == 4  # daily, weekly, monthly, all_time

        # Test with database error
        mock_storage_service.store_performance_metrics.side_effect = DatabaseError("DB Error")
        result = calculator.calculate_metrics()

        assert result is False

    def test_calculate_metrics_no_new_trades(self, calculator, mock_trade_history_service):
        """Test skipping calculation when there are no new trades."""
        # Set last calculation time
        calculator._last_calculation_time = datetime.now()

        # Make get_trades return empty list (no new trades)
        mock_trade_history_service.get_trades.return_value = []

        result = calculator.calculate_metrics()

        assert result is True
        # Should check for new trades but not proceed further
        mock_trade_history_service.get_trades.assert_called_once()

    def test_generate_report(self, calculator, mock_storage_service, monkeypatch):
        """Test generating a performance report."""
        # Set up mock metrics that will be returned by get_performance_metrics
        mock_metrics = {
            "period": "monthly",
            "timestamp": datetime.now().isoformat(),
            "start_balance": 10000.0,
            "end_balance": 10500.0,
            "profit_loss": 500.0,
            "profit_loss_pct": 5.0,
            "total_trades": 3,
            "winning_trades": 2,
            "losing_trades": 1,
            "win_rate": 66.67,
            "max_drawdown": 200.0,
            "max_drawdown_pct": 2.0,
            "avg_profit_winning": 350.0,
            "avg_loss_losing": -200.0,
            "risk_reward_ratio": 1.75,
            "expectancy": 166.67,
        }

        # Configure the mock to return the metrics
        mock_storage_service.get_performance_metrics.return_value = mock_metrics

        # Mock datetime.now() using patch instead of monkeypatch
        first_day = datetime(2023, 6, 1)  # June 1st
        with patch('app.tasks.performance_calculator.datetime') as mock_datetime:
            # Configure the mock to return our fixed date when now() is called
            mock_datetime.now.return_value = first_day
            # Pass through other datetime methods to the real datetime
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            mock_datetime.isoformat = datetime.isoformat
            mock_datetime.fromtimestamp = datetime.fromtimestamp

            report = calculator.generate_report(report_type="monthly")

            # Should have fetched metrics and stored the report
            mock_storage_service.get_performance_metrics.assert_called_once_with(period="monthly")
            mock_storage_service.store_performance_report.assert_called_once()

            # Check that last generated report time was updated
            assert calculator._last_generated_report_time is not None

        # Test skipping monthly report on non-first day
        mock_storage_service.reset_mock()
        calculator._last_generated_report_time = None

        # Mock datetime.now() to return a non-first day
        second_day = datetime(2023, 6, 2)  # June 2nd
        with patch('app.tasks.performance_calculator.datetime') as mock_datetime:
            # Configure the mock to return our fixed date when now() is called
            mock_datetime.now.return_value = second_day
            # Pass through other datetime methods to the real datetime
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            report = calculator.generate_report(report_type="monthly")

            # Should not have generated a report
            assert report == {}
            mock_storage_service.get_performance_metrics.assert_not_called()

        # Test daily report (should generate regardless of day)
        mock_storage_service.reset_mock()

        with patch('app.tasks.performance_calculator.datetime') as mock_datetime:
            # Configure the mock to return our fixed date when now() is called
            mock_datetime.now.return_value = second_day
            # Pass through other datetime methods to the real datetime
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            report = calculator.generate_report(report_type="daily")

            # Should have generated a report
            mock_storage_service.get_performance_metrics.assert_called_once_with(period="daily")
            mock_storage_service.store_performance_report.assert_called_once()

    def test_filter_trades_by_period(self, calculator, mock_trade_history_service):
        """Test filtering trades by time period."""
        trades = mock_trade_history_service.get_trades()

        # Test daily filter (only most recent trade)
        daily_trades = calculator._filter_trades_by_period(trades, "daily")
        assert len(daily_trades) == 1
        assert daily_trades[0]["id"] == "3"

        # Test weekly filter (should include more)
        weekly_trades = calculator._filter_trades_by_period(trades, "weekly")
        assert len(weekly_trades) >= 1

        # Test monthly filter (should include all in our sample)
        monthly_trades = calculator._filter_trades_by_period(trades, "monthly")
        assert len(monthly_trades) == 3

        # Test all_time filter (should include all)
        all_time_trades = calculator._filter_trades_by_period(trades, "all_time")
        assert len(all_time_trades) == 3

    def test_is_winning_trade(self, calculator):
        """Test determining if a trade is winning or losing."""
        winning_trade = {"profit_loss": 100.0}
        losing_trade = {"profit_loss": -50.0}
        breakeven_trade = {"profit_loss": 0.0}

        assert calculator._is_winning_trade(winning_trade) is True
        assert calculator._is_winning_trade(losing_trade) is False
        assert calculator._is_winning_trade(breakeven_trade) is False

    def test_calculate_drawdown(self, calculator):
        """Test calculating maximum drawdown."""
        # Sample trades with a drawdown scenario
        trades = [
            {"timestamp": "2023-01-01T12:00:00", "profit_loss": 100.0},
            {"timestamp": "2023-01-02T12:00:00", "profit_loss": 200.0},
            {"timestamp": "2023-01-03T12:00:00", "profit_loss": -300.0},  # Drawdown here
            {"timestamp": "2023-01-04T12:00:00", "profit_loss": -50.0},   # Continuing drawdown
            {"timestamp": "2023-01-05T12:00:00", "profit_loss": 250.0},   # Recovery
        ]

        start_balance = Decimal("1000")

        max_drawdown, max_drawdown_pct = calculator._calculate_drawdown(trades, start_balance)

        # Max drawdown should be 350 (peak 1300 - trough 950)
        assert max_drawdown == Decimal("350")
        # Max drawdown percentage should be (350/1300)*100 = 26.92%
        assert round(max_drawdown_pct, 2) == Decimal("26.92")

        # Test with empty trades
        max_drawdown, max_drawdown_pct = calculator._calculate_drawdown([], start_balance)
        assert max_drawdown == Decimal("0")
        assert max_drawdown_pct == Decimal("0")

    def test_calculate_average_profit(self, calculator):
        """Test calculating average profit."""
        trades = [
            {"profit_loss": 100.0},
            {"profit_loss": 200.0},
            {"profit_loss": -150.0},
        ]

        # Test average profit from all trades
        avg_profit = calculator._calculate_average_profit(trades)
        assert float(avg_profit) == 50.0

        # Test average profit from winning trades only
        avg_winning = calculator._calculate_average_profit(trades, only_winning=True)
        assert float(avg_winning) == 150.0

        # Test with empty trades
        assert float(calculator._calculate_average_profit([])) == 0.0

    def test_calculate_average_loss(self, calculator):
        """Test calculating average loss."""
        trades = [
            {"profit_loss": 100.0},
            {"profit_loss": 200.0},
            {"profit_loss": -150.0},
        ]

        # Test average loss from losing trades only
        avg_losing = calculator._calculate_average_loss(trades, only_losing=True)
        assert float(avg_losing) == -150.0

        # Test with no losing trades
        winning_trades = [
            {"profit_loss": 100.0},
            {"profit_loss": 200.0},
        ]
        assert float(calculator._calculate_average_loss(winning_trades, only_losing=True)) == 0.0

        # Test with empty trades
        assert float(calculator._calculate_average_loss([])) == 0.0

    def test_build_report(self, calculator, mock_storage_service):
        """Test building a comprehensive report from metrics."""
        metrics = mock_storage_service.get_performance_metrics()

        report = calculator._build_report(metrics, "daily")

        # Check report structure
        assert "report_type" in report
        assert "generated_at" in report
        assert "period_start" in report
        assert "period_end" in report
        assert "summary" in report
        assert "detailed_metrics" in report
        assert "recommendations" in report

        # Check summary content
        summary = report["summary"]
        assert "profit_loss" in summary
        assert "profit_loss_pct" in summary
        assert "total_trades" in summary
        assert "win_rate" in summary
        assert "risk_reward_ratio" in summary

        # Check detailed metrics
        assert report["detailed_metrics"] == metrics

        # Check recommendations
        assert isinstance(report["recommendations"], list)

    def test_generate_recommendations(self, calculator):
        """Test generating recommendations based on metrics."""
        # Test with poor metrics
        poor_metrics = {
            "win_rate": 35.0,
            "risk_reward_ratio": 0.8,
            "max_drawdown_pct": 25.0,
            "profit_loss_pct": -5.0,
        }

        recommendations = calculator._generate_recommendations(poor_metrics)

        # Should have several recommendations for improvement
        assert len(recommendations) >= 3
        assert any("win rate" in rec.lower() for rec in recommendations)
        assert any("risk/reward" in rec.lower() for rec in recommendations)
        assert any("drawdown" in rec.lower() for rec in recommendations)

        # Test with good metrics
        good_metrics = {
            "win_rate": 65.0,
            "risk_reward_ratio": 2.5,
            "max_drawdown_pct": 10.0,
            "profit_loss_pct": 15.0,
        }

        recommendations = calculator._generate_recommendations(good_metrics)

        # Should have a positive recommendation
        assert len(recommendations) == 1
        assert "performing well" in recommendations[0].lower()

    def test_get_period_balances(self, calculator, mock_portfolio_service, mock_config_service):
        """Test getting start and end balances for different periods."""
        current_portfolio = mock_portfolio_service.get_current_portfolio()

        # Test all_time period (should use initial deposit)
        start_balance, end_balance = calculator._get_period_balances("all_time", current_portfolio)
        assert start_balance == Decimal("10000.0")  # Initial deposit
        assert end_balance == Decimal("10500.0")    # Current value

        # Test other periods (should use historical balance)
        for period in ["daily", "weekly", "monthly"]:
            start_balance, end_balance = calculator._get_period_balances(period, current_portfolio)
            assert start_balance == Decimal("10000.0")  # From mock_portfolio_service.get_historical_balance
            assert end_balance == Decimal("10500.0")    # Current value
