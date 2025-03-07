import unittest
from unittest.mock import MagicMock, patch
from decimal import Decimal
from datetime import datetime

from app.models.position import Position
from app.services.position_reporting import PositionReporting
from app.core.database import Database
from app.config.config import Config


class TestPositionReporting(unittest.TestCase):
    """Test cases for the PositionReporting class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MagicMock(spec=Config)

        self.database = MagicMock(spec=Database)
        self.session = MagicMock()
        self.database.get_session.return_value.__enter__.return_value = self.session

        self.logger = MagicMock()

        self.position_reporting = PositionReporting(
            config=self.config,
            database=self.database,
            logger=self.logger
        )

    def test_generate_position_summary(self):
        """Test generating a position summary."""
        # Configure the mock queries
        self.session.query.return_value.scalar.return_value = 10  # total positions
        self.session.query.return_value.filter.return_value.scalar.side_effect = [
            5,  # open positions
            2,  # partially closed positions
            3,  # closed positions
            2,  # winning positions
            1   # losing positions
        ]

        # Configure the profit/loss queries
        self.session.query.return_value.filter.return_value.scalar.side_effect = [
            5,  # open positions
            2,  # partially closed positions
            3,  # closed positions
            Decimal('1500'),  # total profit/loss
            Decimal('12.5'),  # average profit/loss percentage
            2,  # winning positions
            1   # losing positions
        ]

        # Call the method
        summary = self.position_reporting.generate_position_summary()

        # Assertions
        self.assertEqual(summary["total_positions"], 10)
        self.assertEqual(summary["open_positions"], 5)
        self.assertEqual(summary["partially_closed_positions"], 2)
        self.assertEqual(summary["closed_positions"], 3)
        self.assertEqual(summary["total_profit_loss"], Decimal('1500'))
        self.assertEqual(summary["average_profit_loss_percentage"], Decimal('12.5'))
        self.assertEqual(summary["winning_positions"], 2)
        self.assertEqual(summary["losing_positions"], 1)
        self.assertEqual(summary["win_rate"], Decimal('66.66666666666666666666666667'))

        # Verify logger was called
        self.logger.info.assert_called_once()

    def test_calculate_daily_pnl(self):
        """Test calculating daily P&L."""
        # Set up a test date
        test_date = datetime(2023, 5, 15)

        # Configure the mock queries
        self.session.query.return_value.filter.return_value.scalar.return_value = 3  # positions closed today
        self.session.query.return_value.filter.return_value.scalar.side_effect = [
            3,  # positions closed today
            Decimal('750')  # total profit/loss for today
        ]

        # Create mock positions for best/worst
        best_position = MagicMock(spec=Position)
        best_position.id = 1
        best_position.symbol = "BTC/USD"
        best_position.profit_loss = Decimal('500')
        best_position.profit_loss_percentage = Decimal('10')

        worst_position = MagicMock(spec=Position)
        worst_position.id = 2
        worst_position.symbol = "ETH/USD"
        worst_position.profit_loss = Decimal('-100')
        worst_position.profit_loss_percentage = Decimal('-5')

        # Configure the position queries
        self.session.query.return_value.filter.return_value.order_by.return_value.first.side_effect = [
            best_position,  # best position
            worst_position  # worst position
        ]

        # Call the method
        result = self.position_reporting.calculate_daily_pnl(test_date)

        # Assertions
        self.assertEqual(result["date"], "2023-05-15")
        self.assertEqual(result["closed_positions"], 3)
        self.assertEqual(result["total_profit_loss"], Decimal('750'))
        self.assertEqual(result["best_position"]["id"], 1)
        self.assertEqual(result["best_position"]["symbol"], "BTC/USD")
        self.assertEqual(result["best_position"]["profit_loss"], Decimal('500'))
        self.assertEqual(result["best_position"]["profit_loss_percentage"], Decimal('10'))
        self.assertEqual(result["worst_position"]["id"], 2)
        self.assertEqual(result["worst_position"]["symbol"], "ETH/USD")
        self.assertEqual(result["worst_position"]["profit_loss"], Decimal('-100'))
        self.assertEqual(result["worst_position"]["profit_loss_percentage"], Decimal('-5'))

        # Verify logger was called
        self.logger.info.assert_called_once()

    def test_calculate_daily_pnl_no_positions(self):
        """Test calculating daily P&L with no positions."""
        # Set up a test date
        test_date = datetime(2023, 5, 15)

        # Configure the mock queries to return no positions
        self.session.query.return_value.filter.return_value.scalar.return_value = 0
        self.session.query.return_value.filter.return_value.scalar.side_effect = [
            0,  # positions closed today
            Decimal('0')  # total profit/loss for today
        ]

        # Configure the position queries to return None
        self.session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        # Call the method
        result = self.position_reporting.calculate_daily_pnl(test_date)

        # Assertions
        self.assertEqual(result["date"], "2023-05-15")
        self.assertEqual(result["closed_positions"], 0)
        self.assertEqual(result["total_profit_loss"], Decimal('0'))
        self.assertIsNone(result["best_position"])
        self.assertIsNone(result["worst_position"])

    @patch('app.services.position_reporting.PositionReporting.calculate_daily_pnl')
    def test_generate_performance_report(self, mock_calculate_daily_pnl):
        """Test generating a performance report."""
        # Set up test dates
        start_date = datetime(2023, 5, 1)
        end_date = datetime(2023, 5, 15)

        # Create mock positions
        positions = []
        for i in range(10):
            position = MagicMock(spec=Position)
            position.id = i + 1
            position.symbol = "BTC/USD" if i < 7 else "ETH/USD"
            position.strategy_used = "strategy1" if i < 5 else "strategy2"
            position.entry_timestamp = start_date
            position.exit_timestamp = end_date
            position.profit_loss = Decimal(100 * (i - 3))  # Some winners, some losers
            position.profit_loss_percentage = Decimal(10 * (i - 3))
            positions.append(position)

        # Configure the session to return these positions
        self.session.query.return_value.filter.return_value.all.return_value = positions

        # Configure the daily P&L mock
        mock_calculate_daily_pnl.return_value = {
            "date": "2023-05-15",
            "closed_positions": 3,
            "total_profit_loss": Decimal('750')
        }

        # Call the method
        report = self.position_reporting.generate_performance_report(start_date, end_date)

        # Assertions
        self.assertEqual(report["time_period"], "2023-05-01 to 2023-05-15")
        self.assertEqual(report["total_trades"], 10)
        self.assertEqual(report["winning_trades"], 6)
        self.assertEqual(report["losing_trades"], 4)
        self.assertEqual(report["win_rate"], 60.0)

        # Check the profit factor (gross profit / gross loss)
        gross_profit = sum(p.profit_loss for p in positions if p.profit_loss > 0)
        gross_loss = abs(sum(p.profit_loss for p in positions if p.profit_loss < 0))
        expected_profit_factor = gross_profit / gross_loss
        self.assertEqual(report["profit_factor"], expected_profit_factor)

        # Check largest winner and loser
        largest_winner = max(positions, key=lambda p: p.profit_loss)
        largest_loser = min(positions, key=lambda p: p.profit_loss)
        self.assertEqual(report["largest_winner"]["id"], largest_winner.id)
        self.assertEqual(report["largest_loser"]["id"], largest_loser.id)

        # Check symbol and strategy breakdowns
        self.assertIn("BTC/USD", report["symbols"])
        self.assertIn("ETH/USD", report["symbols"])
        self.assertIn("strategy1", report["strategies"])
        self.assertIn("strategy2", report["strategies"])

        # Verify logger was called
        self.logger.info.assert_called_once()

    def test_generate_performance_report_no_trades(self):
        """Test generating a performance report with no trades."""
        # Set up test dates
        start_date = datetime(2023, 5, 1)
        end_date = datetime(2023, 5, 15)

        # Configure the session to return no positions
        self.session.query.return_value.filter.return_value.all.return_value = []

        # Call the method
        report = self.position_reporting.generate_performance_report(start_date, end_date)

        # Assertions
        self.assertEqual(report["time_period"], "2023-05-01 to 2023-05-15")
        self.assertEqual(report["total_trades"], 0)
        self.assertIn("message", report)
        self.assertEqual(report["message"], "No closed trades in this period")

        # Verify logger was called
        self.logger.info.assert_called_once()
