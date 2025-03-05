import pytest
from decimal import Decimal
from datetime import datetime
import logging

from app.strategies.base_strategy import BaseStrategy
from app.models.position import PositionType, PositionStatus


# Define a concrete implementation of BaseStrategy for testing
class ConcreteTestStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.description = "Test strategy for unit testing"

    def generate_signals(self, market_data):
        """Test implementation of generate_signals."""
        return [{"symbol": "BTC-USD", "type": "BUY", "confidence": 0.8}]

    def should_enter_position(self, symbol, market_data):
        """Test implementation of should_enter_position."""
        return True, {"reason": "Test reason"}

    def should_exit_position(self, position, market_data):
        """Test implementation of should_exit_position."""
        return False, {"reason": "Test reason"}

    def calculate_position_size(self, symbol, account_balance):
        """Test implementation of calculate_position_size."""
        return account_balance * Decimal("0.1")

    def get_stop_loss(self, symbol, entry_price, market_data):
        """Test implementation of get_stop_loss."""
        return entry_price * Decimal("0.95")

    def get_take_profit(self, symbol, entry_price, market_data):
        """Test implementation of get_take_profit."""
        return entry_price * Decimal("1.1")


class TestBaseStrategy:
    """Tests for the BaseStrategy class."""

    @pytest.fixture
    def valid_config(self):
        """Return a valid configuration for testing."""
        return {
            "risk_per_trade": 2,
            "max_open_positions": 5,
            "timeframes": ["1m", "5m", "1h"],
            "indicators": ["ema_50", "rsi_14"]
        }

    @pytest.fixture
    def invalid_config(self):
        """Return an invalid configuration for testing."""
        return {
            "risk_per_trade": 10,  # Invalid: too high
            "max_open_positions": 0  # Invalid: too low
        }

    @pytest.fixture
    def test_strategy(self, valid_config):
        """Return a test strategy instance."""
        return ConcreteTestStrategy(valid_config)

    @pytest.fixture
    def sample_position(self):
        """Return a sample position for testing."""
        # Create a mock Position object instead of an actual database model
        class MockPosition:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return MockPosition(
            symbol="BTC-USD",
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            position_type=PositionType.LONG,
            stop_loss_price=Decimal("48000"),
            take_profit_price=Decimal("55000"),
            status=PositionStatus.OPEN,
            entry_timestamp=datetime.utcnow(),
            strategy_used="TestStrategy"
        )

    def test_initialization_with_valid_config(self, valid_config):
        """Test initialization with a valid configuration."""
        strategy = ConcreteTestStrategy(valid_config)
        assert strategy.name == "ConcreteTestStrategy"
        assert strategy.description == "Test strategy for unit testing"
        assert strategy.config == valid_config
        assert isinstance(strategy.logger, logging.Logger)

    def test_initialization_with_invalid_config(self, invalid_config):
        """Test initialization with an invalid configuration."""
        with pytest.raises(ValueError):
            ConcreteTestStrategy(invalid_config)

    def test_validate_config(self, test_strategy, valid_config):
        """Test the validate_config method."""
        # This should not raise an exception
        test_strategy.validate_config()

        # Test with missing required parameter
        test_strategy.config.pop("risk_per_trade")
        with pytest.raises(ValueError):
            test_strategy.validate_config()

    def test_calculate_risk_amount(self, test_strategy):
        """Test the calculate_risk_amount method."""
        account_balance = Decimal("10000")
        expected_risk = account_balance * Decimal("0.02")  # 2% risk

        risk_amount = test_strategy.calculate_risk_amount(account_balance)

        assert risk_amount == expected_risk

    def test_calculate_risk_reward_ratio_long(self, test_strategy):
        """Test the calculate_risk_reward_ratio method for long positions."""
        entry_price = Decimal("50000")
        stop_loss = Decimal("48000")
        take_profit = Decimal("55000")

        # For a long position: risk = entry - stop, reward = take - entry
        expected_rr = (Decimal("55000") - Decimal("50000")) / (Decimal("50000") - Decimal("48000"))

        risk_reward = test_strategy.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)

        assert risk_reward == expected_rr

    def test_calculate_risk_reward_ratio_short(self, test_strategy):
        """Test the calculate_risk_reward_ratio method for short positions."""
        entry_price = Decimal("50000")
        stop_loss = Decimal("52000")
        take_profit = Decimal("45000")

        # For a short position: risk = stop - entry, reward = entry - take
        expected_rr = (Decimal("50000") - Decimal("45000")) / (Decimal("52000") - Decimal("50000"))

        risk_reward = test_strategy.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)

        assert risk_reward == expected_rr

    def test_calculate_risk_reward_ratio_invalid(self, test_strategy):
        """Test the calculate_risk_reward_ratio method with invalid prices."""
        # Invalid setup: entry below stop loss but take profit above entry (mixed long/short)
        with pytest.raises(ValueError):
            test_strategy.calculate_risk_reward_ratio(
                Decimal("50000"), Decimal("52000"), Decimal("55000")
            )

        # Zero risk
        with pytest.raises(ValueError):
            test_strategy.calculate_risk_reward_ratio(
                Decimal("50000"), Decimal("50000"), Decimal("55000")
            )

    def test_is_valid_trade_setup(self, test_strategy):
        """Test the is_valid_trade_setup method."""
        # Valid setup with RR > 1.5
        assert test_strategy.is_valid_trade_setup(
            Decimal("50000"), Decimal("48000"), Decimal("55000")
        )

        # Invalid setup with RR < 1.5
        assert not test_strategy.is_valid_trade_setup(
            Decimal("50000"), Decimal("48000"), Decimal("52000")
        )

        # Invalid price configuration
        assert not test_strategy.is_valid_trade_setup(
            Decimal("50000"), Decimal("52000"), Decimal("55000")
        )

    def test_get_timeframes(self, test_strategy):
        """Test the get_timeframes method."""
        expected_timeframes = ["1m", "5m", "1h"]
        assert test_strategy.get_timeframes() == expected_timeframes

    def test_get_required_indicators(self, test_strategy):
        """Test the get_required_indicators method."""
        expected_indicators = ["ema_50", "rsi_14"]
        assert test_strategy.get_required_indicators() == expected_indicators

    def test_update_config(self, test_strategy):
        """Test the update_config method."""
        new_config = {"risk_per_trade": 3, "timeframes": ["15m", "1h", "4h"]}

        test_strategy.update_config(new_config)

        assert test_strategy.config["risk_per_trade"] == 3
        assert test_strategy.config["timeframes"] == ["15m", "1h", "4h"]
        assert test_strategy.config["max_open_positions"] == 5  # Unchanged

        # Test with invalid update
        with pytest.raises(ValueError):
            test_strategy.update_config({"risk_per_trade": 10})  # Too high

    def test_abstract_methods_implementation(self, test_strategy, sample_position):
        """Test that the concrete implementation properly implements abstract methods."""
        market_data = {"BTC-USD": {"price": Decimal("51000")}}

        # Test generate_signals
        signals = test_strategy.generate_signals(market_data)
        assert isinstance(signals, list)
        assert len(signals) == 1
        assert signals[0]["symbol"] == "BTC-USD"

        # Test should_enter_position
        should_enter, info = test_strategy.should_enter_position("BTC-USD", market_data)
        assert isinstance(should_enter, bool)
        assert isinstance(info, dict)

        # Test should_exit_position
        should_exit, info = test_strategy.should_exit_position(sample_position, market_data)
        assert isinstance(should_exit, bool)
        assert isinstance(info, dict)

        # Test calculate_position_size
        position_size = test_strategy.calculate_position_size("BTC-USD", Decimal("10000"))
        assert isinstance(position_size, Decimal)

        # Test get_stop_loss
        stop_loss = test_strategy.get_stop_loss("BTC-USD", Decimal("50000"), market_data)
        assert isinstance(stop_loss, Decimal)

        # Test get_take_profit
        take_profit = test_strategy.get_take_profit("BTC-USD", Decimal("50000"), market_data)
        assert isinstance(take_profit, (Decimal, list))
