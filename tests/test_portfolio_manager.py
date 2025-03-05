import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import datetime

from app.services.portfolio_manager import PortfolioManager
from app.models.position import Position, PositionType, PositionStatus
from app.config.config import Config


class TestPortfolioManager:
    """Tests for the PortfolioManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with portfolio management settings."""
        config = MagicMock(spec=Config)

        # Set up the portfolio management config
        portfolio_config = {
            "account_balance": 10000.0,  # $10,000 account
            "max_positions": 10,
            "max_symbols": 5,
            "max_positions_per_symbol": 2,
            "max_exposure": {
                "default": 5.0,  # 5% max exposure per symbol
                "BTC/USDT": 10.0  # 10% max for BTC
            },
            "prevent_opposite_positions": True,
            "max_total_exposure": 50.0,  # 50% max total exposure
            "max_exposure_ratio": 80.0,  # 80% max exposure ratio
            "max_symbol_concentration": 30.0  # 30% max concentration per symbol
        }

        config.get_nested.return_value = portfolio_config
        return config

    @pytest.fixture
    def portfolio_manager(self, mock_config):
        """Create a PortfolioManager instance with mock config."""
        return PortfolioManager(mock_config)

    @pytest.fixture
    def sample_position(self):
        """Create a sample position for testing."""
        position = Position()
        position.id = "test-position-1"
        position.symbol = "BTC/USDT"
        position.entry_price = Decimal("50000")
        position.size = Decimal("0.1")  # 0.1 BTC
        position.position_type = PositionType.LONG
        position.stop_loss_price = Decimal("49000")
        position.take_profit_price = Decimal("52000")
        position.status = PositionStatus.OPEN
        position.entry_timestamp = datetime.utcnow()
        return position

    @pytest.fixture
    def sample_positions(self, sample_position):
        """Create a list of sample positions for testing."""
        positions = [sample_position]

        # Add ETH position
        eth_position = Position()
        eth_position.id = "test-position-2"
        eth_position.symbol = "ETH/USDT"
        eth_position.entry_price = Decimal("3000")
        eth_position.size = Decimal("2")  # 2 ETH
        eth_position.position_type = PositionType.LONG
        eth_position.stop_loss_price = Decimal("2900")
        eth_position.take_profit_price = Decimal("3200")
        eth_position.status = PositionStatus.OPEN
        eth_position.entry_timestamp = datetime.utcnow()
        positions.append(eth_position)

        # Add another BTC position
        btc_position2 = Position()
        btc_position2.id = "test-position-3"
        btc_position2.symbol = "BTC/USDT"
        btc_position2.entry_price = Decimal("51000")
        btc_position2.size = Decimal("0.05")  # 0.05 BTC
        btc_position2.position_type = PositionType.LONG
        btc_position2.stop_loss_price = Decimal("50000")
        btc_position2.take_profit_price = Decimal("53000")
        btc_position2.status = PositionStatus.OPEN
        btc_position2.entry_timestamp = datetime.utcnow()
        positions.append(btc_position2)

        return positions

    def test_add_position(self, portfolio_manager, sample_position):
        """Test adding a position to the portfolio."""
        # Mock the check_position_limits method to always return True
        original_check = portfolio_manager.check_position_limits
        portfolio_manager.check_position_limits = lambda x: (True, None)

        try:
            # Add the position
            position_id = portfolio_manager.add_position(sample_position)

            # Verify it was added
            assert position_id == sample_position.id
            assert position_id in portfolio_manager.positions
            assert portfolio_manager.positions[position_id] == sample_position

            # Test adding a position without an ID
            position = Position()
            position.symbol = "ETH/USDT"
            position.entry_price = Decimal("3000")
            position.size = Decimal("1")
            position.position_type = PositionType.LONG

            position_id = portfolio_manager.add_position(position)

            # Verify an ID was generated and position was added
            assert position_id is not None
            assert position_id in portfolio_manager.positions
            assert portfolio_manager.positions[position_id].symbol == "ETH/USDT"

            # Test adding a position with missing required attributes
            invalid_position = Position()

            with pytest.raises(ValueError, match="Position missing required attributes"):
                portfolio_manager.add_position(invalid_position)
        finally:
            # Restore the original method
            portfolio_manager.check_position_limits = original_check

    def test_update_position(self, portfolio_manager, sample_position):
        """Test updating an existing position."""
        # Add the position directly to the positions dict
        portfolio_manager.positions[sample_position.id] = sample_position

        # Update the position
        updates = {
            "stop_loss_price": Decimal("48500"),
            "take_profit_price": Decimal("53000")
        }

        updated_position = portfolio_manager.update_position(sample_position.id, updates)

        # Verify updates were applied
        assert updated_position.stop_loss_price == Decimal("48500")
        assert updated_position.take_profit_price == Decimal("53000")

        # Verify the position in the manager was updated
        assert portfolio_manager.positions[sample_position.id].stop_loss_price == Decimal("48500")

        # Test updating a non-existent position
        with pytest.raises(KeyError, match="Position with ID .* not found"):
            portfolio_manager.update_position("non-existent-id", updates)

    def test_close_position_full(self, portfolio_manager, sample_position):
        """Test fully closing a position."""
        # Add the position directly to the positions dict
        portfolio_manager.positions[sample_position.id] = sample_position

        # Close the position
        exit_price = Decimal("51000")
        closed_position = portfolio_manager.close_position(sample_position.id, exit_price)

        # Verify position was closed
        assert closed_position.status == PositionStatus.CLOSED
        assert closed_position.exit_price == exit_price
        assert closed_position.exit_timestamp is not None

        # Verify position was removed from active tracking
        assert sample_position.id not in portfolio_manager.positions

        # Test closing a non-existent position
        with pytest.raises(KeyError, match="Position with ID .* not found"):
            portfolio_manager.close_position("non-existent-id", exit_price)

    def test_close_position_partial(self, portfolio_manager, sample_position):
        """Test partially closing a position."""
        # Add the position directly to the positions dict
        portfolio_manager.positions[sample_position.id] = sample_position

        # Mock the apply_exit method to simulate partial exit
        def mock_apply_exit(price, timestamp, full_exit, exit_percentage=None):
            if not full_exit:
                # Simulate partial exit
                sample_position.size = sample_position.size * (Decimal('100') - exit_percentage) / Decimal('100')
                sample_position.status = PositionStatus.PARTIALLY_CLOSED
                return True
            return None

        sample_position.apply_exit = mock_apply_exit

        # Partially close the position
        exit_price = Decimal("51000")
        exit_percentage = Decimal("50")

        updated_position = portfolio_manager.close_position(
            sample_position.id,
            exit_price,
            partial_exit=True,
            exit_percentage=exit_percentage
        )

        # Verify position was partially closed
        assert updated_position.status == PositionStatus.PARTIALLY_CLOSED
        assert updated_position.size == Decimal("0.05")  # 50% of 0.1

        # Verify position is still being tracked
        assert sample_position.id in portfolio_manager.positions

        # Test partial exit with invalid percentage
        with pytest.raises(ValueError, match="Valid exit_percentage required for partial exit"):
            portfolio_manager.close_position(
                sample_position.id,
                exit_price,
                partial_exit=True,
                exit_percentage=Decimal("0")
            )

    def test_get_current_exposure(self, portfolio_manager, sample_positions):
        """Test getting current portfolio exposure metrics."""
        # Load the sample positions
        portfolio_manager.load_positions(sample_positions)

        # Get exposure metrics
        exposure = portfolio_manager.get_current_exposure()

        # Expected:
        # Total exposure = 0.1 + 2 + 0.05 = 2.15
        # Exposure per symbol = {"BTC/USDT": 0.15, "ETH/USDT": 2}
        # Symbols count = 2
        # Positions count = 3
        # Exposure ratio = (2.15 / 10000) * 100 = 0.0215%
        assert exposure["total_exposure"] == Decimal("2.15")
        assert exposure["exposure_per_symbol"]["BTC/USDT"] == Decimal("0.15")
        assert exposure["exposure_per_symbol"]["ETH/USDT"] == Decimal("2")
        assert exposure["symbols_count"] == 2
        assert exposure["positions_count"] == 3
        assert exposure["exposure_ratio"] == Decimal("0.0215")

    def test_get_exposure_per_symbol(self, portfolio_manager, sample_positions):
        """Test getting exposure metrics for a specific symbol."""
        # Load the sample positions
        portfolio_manager.load_positions(sample_positions)

        # Get BTC exposure metrics
        btc_exposure = portfolio_manager.get_exposure_per_symbol("BTC/USDT")

        # Expected:
        # Total exposure = 0.1 + 0.05 = 0.15
        # Long exposure = 0.15
        # Short exposure = 0
        # Net exposure = 0.15
        # Positions count = 2
        # Ratio of portfolio = (0.15 / 2.15) * 100 = 6.98%
        assert btc_exposure["total_exposure"] == Decimal("0.15")
        assert btc_exposure["long_exposure"] == Decimal("0.15")
        assert btc_exposure["short_exposure"] == Decimal("0")
        assert btc_exposure["net_exposure"] == Decimal("0.15")
        assert btc_exposure["positions_count"] == 2
        assert round(btc_exposure["ratio_of_portfolio"], 2) == Decimal("6.98")

        # Get exposure for a symbol with no positions
        xrp_exposure = portfolio_manager.get_exposure_per_symbol("XRP/USDT")

        # Expected: All zeros
        assert xrp_exposure["total_exposure"] == Decimal("0")
        assert xrp_exposure["positions_count"] == 0

    def test_check_position_limits(self, portfolio_manager, sample_positions):
        """Test checking if a new position would exceed portfolio limits."""
        # Load the sample positions
        portfolio_manager.load_positions(sample_positions)

        # Test valid position
        new_position = Position()
        new_position.symbol = "XRP/USDT"
        new_position.size = Decimal("0.5")  # Use a small position size that won't exceed limits
        new_position.position_type = PositionType.LONG

        valid, reason = portfolio_manager.check_position_limits(new_position)
        assert valid is True
        assert reason is None

        # Test exceeding max positions per symbol
        portfolio_manager._portfolio_config["max_positions_per_symbol"] = 2

        new_position = Position()
        new_position.symbol = "BTC/USDT"  # Already have 2 BTC positions
        new_position.size = Decimal("0.1")
        new_position.position_type = PositionType.LONG

        valid, reason = portfolio_manager.check_position_limits(new_position)
        assert valid is False
        assert "Maximum positions for BTC/USDT reached" in reason

        # Reset max positions per symbol
        portfolio_manager._portfolio_config["max_positions_per_symbol"] = 5

        # Test exceeding symbol exposure
        new_position = Position()
        new_position.symbol = "BTC/USDT"
        new_position.size = Decimal("1.0")  # Would exceed 10% max for BTC
        new_position.position_type = PositionType.LONG

        valid, reason = portfolio_manager.check_position_limits(new_position)
        assert valid is False
        assert "Maximum concentration for BTC/USDT would be exceeded" in reason

        # Test opposite direction position
        new_position = Position()
        new_position.symbol = "BTC/USDT"
        new_position.size = Decimal("0.1")
        new_position.position_type = PositionType.SHORT  # Opposite of existing LONG

        valid, reason = portfolio_manager.check_position_limits(new_position)
        assert valid is False
        assert "Position in opposite direction already exists" in reason

        # Test maximum concentration
        new_position = Position()
        new_position.symbol = "LTC/USDT"
        new_position.size = Decimal("10")  # Would make LTC > 30% of portfolio
        new_position.position_type = PositionType.LONG

        valid, reason = portfolio_manager.check_position_limits(new_position)
        assert valid is False
        assert "Maximum concentration for LTC/USDT would be exceeded" in reason

    def test_get_all_positions(self, portfolio_manager, sample_positions):
        """Test getting all tracked positions."""
        # Load the sample positions
        portfolio_manager.load_positions(sample_positions)

        # Get all positions
        positions = portfolio_manager.get_all_positions()

        # Verify
        assert len(positions) == 3
        assert set(p.id for p in positions) == set(p.id for p in sample_positions)

    def test_get_position(self, portfolio_manager, sample_position):
        """Test getting a specific position by ID."""
        # Add the position directly to the positions dict
        portfolio_manager.positions[sample_position.id] = sample_position

        # Get the position
        position = portfolio_manager.get_position(sample_position.id)

        # Verify
        assert position == sample_position

        # Test getting a non-existent position
        with pytest.raises(KeyError, match="Position with ID .* not found"):
            portfolio_manager.get_position("non-existent-id")

    def test_get_positions_by_symbol(self, portfolio_manager, sample_positions):
        """Test getting positions for a specific symbol."""
        # Load the sample positions
        portfolio_manager.load_positions(sample_positions)

        # Get BTC positions
        btc_positions = portfolio_manager.get_positions_by_symbol("BTC/USDT")

        # Verify
        assert len(btc_positions) == 2
        assert all(p.symbol == "BTC/USDT" for p in btc_positions)

        # Get positions for a symbol with no positions
        xrp_positions = portfolio_manager.get_positions_by_symbol("XRP/USDT")

        # Verify
        assert len(xrp_positions) == 0

    def test_get_positions_by_type(self, portfolio_manager, sample_positions):
        """Test getting positions of a specific type."""
        # Load the sample positions
        portfolio_manager.load_positions(sample_positions)

        # Get LONG positions
        long_positions = portfolio_manager.get_positions_by_type(PositionType.LONG)

        # Verify
        assert len(long_positions) == 3
        assert all(p.position_type == PositionType.LONG for p in long_positions)

        # Get SHORT positions (none in our sample)
        short_positions = portfolio_manager.get_positions_by_type(PositionType.SHORT)

        # Verify
        assert len(short_positions) == 0

    def test_load_positions(self, portfolio_manager, sample_positions):
        """Test loading multiple positions at once."""
        # Load the positions
        portfolio_manager.load_positions(sample_positions)

        # Verify all positions were loaded
        assert len(portfolio_manager.positions) == 3
        assert set(p.id for p in sample_positions) == set(portfolio_manager.positions.keys())

        # Test loading a position without an ID
        position = Position()
        position.symbol = "XRP/USDT"
        position.entry_price = Decimal("1.0")
        position.size = Decimal("1000")
        position.position_type = PositionType.LONG

        portfolio_manager.load_positions([position])

        # Verify an ID was generated and position was added
        assert len(portfolio_manager.positions) == 4
        assert any(p.symbol == "XRP/USDT" for p in portfolio_manager.positions.values())
