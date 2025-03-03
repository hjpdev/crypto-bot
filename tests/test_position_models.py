import unittest
from decimal import Decimal
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.base_model import Base
from app.models.cryptocurrency import Cryptocurrency
from app.models.position import Position, PositionType, PositionStatus
from app.core.exceptions import ValidationError


class TestPositionModels(unittest.TestCase):
    """Test case for Position and PartialExit models."""

    @classmethod
    def setUpClass(cls):
        """Set up the test database and tables."""
        cls.engine = create_engine('sqlite:///:memory:')
        cls.SessionLocal = sessionmaker(bind=cls.engine)
        Base.metadata.create_all(cls.engine)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        Base.metadata.drop_all(cls.engine)

    def setUp(self):
        """Create a new session for each test."""
        self.session = self.SessionLocal()

        # Drop and recreate tables for each test to ensure a clean state
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        # Create test cryptocurrency with a unique symbol per test
        test_id = id(self)  # Use object id to create unique symbols for each test instance
        self.crypto = Cryptocurrency(
            symbol=f"BTC/USD-{test_id}",
            name="Bitcoin",
            is_active=True
        )
        self.session.add(self.crypto)
        self.session.commit()

        # Create a sample position
        self.long_position = Position(
            cryptocurrency_id=self.crypto.id,
            symbol=self.crypto.symbol,
            entry_timestamp=datetime.utcnow() - timedelta(days=1),
            entry_price=Decimal('50000.00'),
            size=Decimal('1.0'),
            position_type=PositionType.LONG,
            stop_loss_price=Decimal('45000.00'),
            take_profit_price=Decimal('60000.00'),
            status=PositionStatus.OPEN,
            strategy_used="Test Strategy"
        )
        self.session.add(self.long_position)

        self.short_position = Position(
            cryptocurrency_id=self.crypto.id,
            symbol=self.crypto.symbol,
            entry_timestamp=datetime.utcnow() - timedelta(days=1),
            entry_price=Decimal('50000.00'),
            size=Decimal('1.0'),
            position_type=PositionType.SHORT,
            stop_loss_price=Decimal('55000.00'),
            take_profit_price=Decimal('40000.00'),
            status=PositionStatus.OPEN,
            strategy_used="Test Strategy"
        )
        self.session.add(self.short_position)
        self.session.commit()

    def tearDown(self):
        """Clean up the session after each test."""
        self.session.close()

    def test_create_position(self):
        """Test creating a position."""
        position = Position(
            cryptocurrency_id=self.crypto.id,
            symbol=self.crypto.symbol,
            entry_timestamp=datetime.utcnow(),
            entry_price=Decimal('51000.00'),
            size=Decimal('0.5'),
            position_type=PositionType.LONG,
            stop_loss_price=Decimal('49000.00'),
            take_profit_price=Decimal('55000.00'),
            status=PositionStatus.OPEN,
            strategy_used="RSI Strategy",
            notes="Test position for RSI strategy"
        )
        self.session.add(position)
        self.session.commit()

        # Verify the position was saved to the database
        retrieved_position = self.session.query(Position).filter_by(id=position.id).first()
        self.assertIsNotNone(retrieved_position)
        self.assertEqual(retrieved_position.size, Decimal('0.5'))
        self.assertEqual(retrieved_position.position_type, PositionType.LONG)
        self.assertEqual(retrieved_position.status, PositionStatus.OPEN)
        self.assertEqual(retrieved_position.strategy_used, "RSI Strategy")

        # Test relationship to cryptocurrency
        self.assertRegex(retrieved_position.cryptocurrency.symbol, "BTC/USD")

    def test_calculate_current_pl_long(self):
        """Test calculating P&L for a long position."""
        # Test profit scenario
        profit_price = Decimal('55000.00')
        pl, pl_percentage = self.long_position.calculate_current_pl(profit_price)

        expected_pl = (profit_price - self.long_position.entry_price) * self.long_position.size
        expected_pl_percentage = ((profit_price / self.long_position.entry_price) - 1) * 100

        self.assertEqual(pl, expected_pl)
        self.assertEqual(pl_percentage, expected_pl_percentage)

        # Test loss scenario
        loss_price = Decimal('48000.00')
        pl, pl_percentage = self.long_position.calculate_current_pl(loss_price)

        expected_pl = (loss_price - self.long_position.entry_price) * self.long_position.size
        expected_pl_percentage = ((loss_price / self.long_position.entry_price) - 1) * 100

        self.assertEqual(pl, expected_pl)
        self.assertEqual(pl_percentage, expected_pl_percentage)

    def test_calculate_current_pl_short(self):
        """Test calculating P&L for a short position."""
        # Test profit scenario
        profit_price = Decimal('45000.00')
        pl, pl_percentage = self.short_position.calculate_current_pl(profit_price)

        expected_pl = (self.short_position.entry_price - profit_price) * self.short_position.size
        expected_pl_percentage = ((self.short_position.entry_price / profit_price) - 1) * 100

        self.assertEqual(pl, expected_pl)
        self.assertEqual(pl_percentage, expected_pl_percentage)

        # Test loss scenario
        loss_price = Decimal('52000.00')
        pl, pl_percentage = self.short_position.calculate_current_pl(loss_price)

        expected_pl = (self.short_position.entry_price - loss_price) * self.short_position.size
        expected_pl_percentage = ((self.short_position.entry_price / loss_price) - 1) * 100

        self.assertEqual(pl, expected_pl)
        self.assertEqual(pl_percentage, expected_pl_percentage)

    def test_should_exit_long(self):
        """Test should_exit logic for a long position."""
        # Should not exit when price is between SL and TP
        self.assertFalse(self.long_position.should_exit(Decimal('50000.00')))

        # Should exit at stop loss
        self.assertTrue(self.long_position.should_exit(Decimal('45000.00')))
        self.assertTrue(self.long_position.should_exit(Decimal('44999.99')))

        # Should exit at take profit
        self.assertTrue(self.long_position.should_exit(Decimal('60000.00')))
        self.assertTrue(self.long_position.should_exit(Decimal('60000.01')))

    def test_should_exit_short(self):
        """Test should_exit logic for a short position."""
        # Should not exit when price is between SL and TP
        self.assertFalse(self.short_position.should_exit(Decimal('50000.00')))

        # Should exit at stop loss
        self.assertTrue(self.short_position.should_exit(Decimal('55000.00')))
        self.assertTrue(self.short_position.should_exit(Decimal('55000.01')))

        # Should exit at take profit
        self.assertTrue(self.short_position.should_exit(Decimal('40000.00')))
        self.assertTrue(self.short_position.should_exit(Decimal('39999.99')))

    def test_full_exit(self):
        """Test full exit from a position."""
        # Apply a full exit to the long position
        exit_price = Decimal('52000.00')
        exit_time = datetime.utcnow()

        self.long_position.apply_exit(exit_price, exit_time, full_exit=True)

        # Verify the position is now closed
        self.assertEqual(self.long_position.status, PositionStatus.CLOSED)
        self.assertEqual(self.long_position.exit_price, exit_price)
        self.assertEqual(self.long_position.exit_timestamp, exit_time)

        # Verify P&L calculation
        expected_pl = (exit_price - self.long_position.entry_price) * self.long_position.size
        expected_pl_percentage = ((exit_price / self.long_position.entry_price) - 1) * 100

        self.assertEqual(self.long_position.profit_loss, expected_pl)
        self.assertEqual(self.long_position.profit_loss_percentage, expected_pl_percentage)

    def test_partial_exit(self):
        """Test partial exit from a position."""
        # Apply a partial exit of 50%
        exit_price = Decimal('52000.00')
        exit_time = datetime.utcnow()
        exit_percentage = Decimal('50.0')

        partial_exit = self.long_position.apply_exit(
            exit_price,
            exit_time,
            full_exit=False,
            exit_percentage=exit_percentage
        )

        # Verify the partial exit record
        self.assertIsNotNone(partial_exit)
        self.assertEqual(partial_exit.exit_price, exit_price)
        self.assertEqual(partial_exit.exit_timestamp, exit_time)
        self.assertEqual(partial_exit.exit_percentage, exit_percentage)
        self.assertFalse(partial_exit.trailing_stop_activated)

        # Verify the position is partially closed
        self.assertEqual(self.long_position.status, PositionStatus.PARTIALLY_CLOSED)

        # Save the partial exit
        self.session.add(partial_exit)
        self.session.commit()

        # Verify relationship
        self.assertEqual(len(self.long_position.partial_exits), 1)
        self.assertEqual(self.long_position.partial_exits[0].exit_percentage, exit_percentage)

        # Test another partial exit
        second_exit = self.long_position.apply_exit(
            Decimal('53000.00'),
            datetime.utcnow(),
            full_exit=False,
            exit_percentage=Decimal('25.0')
        )

        self.session.add(second_exit)
        self.session.commit()

        # Verify we now have two partial exits
        self.assertEqual(len(self.long_position.partial_exits), 2)

        # Calculate current P&L with partial exits
        current_price = Decimal('54000.00')
        pl, pl_percentage = self.long_position.calculate_current_pl(current_price)

        # Only 25% of the position remains
        remaining_size = self.long_position.size * Decimal('0.25')
        expected_pl = (current_price - self.long_position.entry_price) * remaining_size

        # The percentage gain is independent of the position size
        expected_pl_percentage = ((current_price / self.long_position.entry_price) - 1) * 100

        self.assertAlmostEqual(pl, expected_pl, places=2)
        self.assertAlmostEqual(pl_percentage, expected_pl_percentage, places=2)

    def test_exit_validation(self):
        """Test validation during exits."""
        exit_time = datetime.utcnow()

        # Test validation for partial exit percentage
        with self.assertRaises(ValidationError):
            self.long_position.apply_exit(
                Decimal('52000.00'),
                exit_time,
                full_exit=False,
                exit_percentage=Decimal('0.0')  # Invalid percentage
            )

        with self.assertRaises(ValidationError):
            self.long_position.apply_exit(
                Decimal('52000.00'),
                exit_time,
                full_exit=False,
                exit_percentage=Decimal('100.0')  # Invalid percentage (must be < 100)
            )

        # Can't exit a closed position
        self.long_position.status = PositionStatus.CLOSED
        with self.assertRaises(ValidationError):
            self.long_position.apply_exit(
                Decimal('52000.00'),
                exit_time,
                full_exit=True
            )

    def test_get_open_positions(self):
        """Test retrieving open positions."""
        # Close one position
        self.long_position.apply_exit(
            Decimal('52000.00'),
            datetime.utcnow(),
            full_exit=True
        )
        self.session.commit()

        # Get open positions
        open_positions = Position.get_open_positions(self.session)

        # Should be only one open position (the short position)
        self.assertEqual(len(open_positions), 1)
        self.assertEqual(open_positions[0].id, self.short_position.id)

        # Partially close the short position
        self.short_position.apply_exit(
            Decimal('48000.00'),
            datetime.utcnow(),
            full_exit=False,
            exit_percentage=Decimal('50.0')
        )
        self.session.commit()

        # Partially closed positions should still count as "open"
        open_positions = Position.get_open_positions(self.session)
        self.assertEqual(len(open_positions), 1)
        self.assertEqual(open_positions[0].status, PositionStatus.PARTIALLY_CLOSED)

    def test_cryptocurrency_relationship(self):
        """Test relationship between Position and Cryptocurrency."""
        crypto_positions = self.crypto.positions
        self.assertEqual(len(crypto_positions), 2)

        # Use a unique symbol for the new cryptocurrency
        test_id = id(self)
        new_crypto = Cryptocurrency(
            symbol=f"ETH/USD-{test_id}",
            name="Ethereum",
            is_active=True
        )
        self.session.add(new_crypto)
        self.session.commit()

        new_position = Position(
            cryptocurrency_id=new_crypto.id,
            symbol=new_crypto.symbol,
            entry_timestamp=datetime.utcnow(),
            entry_price=Decimal('3000.00'),
            size=Decimal('10.0'),
            position_type=PositionType.LONG,
            stop_loss_price=Decimal('2800.00'),
            take_profit_price=Decimal('3500.00'),
            status=PositionStatus.OPEN,
            strategy_used="ETH Strategy"
        )
        self.session.add(new_position)
        self.session.commit()

        self.assertEqual(new_position.cryptocurrency.symbol, f"ETH/USD-{test_id}")
        self.assertEqual(len(new_crypto.positions), 1)


if __name__ == '__main__':
    unittest.main()
