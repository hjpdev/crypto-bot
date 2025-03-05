"""
Tests for the Momentum Strategy implementation.

This module contains tests for the MomentumStrategy class, including
signal generation, entry/exit logic, position sizing, and risk management.
"""

import unittest
from unittest.mock import patch, MagicMock
from decimal import Decimal
import pandas as pd
import numpy as np

from app.strategies.momentum_strategy import MomentumStrategy
from app.models.position import Position, PositionType
from app.models.signals import SignalType
from app.services.indicator_service import IndicatorService


class TestMomentumStrategy(unittest.TestCase):
    """Test cases for the MomentumStrategy class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a sample configuration for the strategy
        self.config = {
            "risk_per_trade": 2.0,
            "max_open_positions": 5,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "volume_change_threshold": 1.5,
            "trend_ema_period": 50,
            "min_confidence_threshold": 0.6,
            "atr_multiplier": 2.0,
            "risk_reward_targets": [1.5, 2.5, 3.5],
            "timeframes": ["15m", "1h", "4h"]
        }

        # Initialize the strategy with the configuration
        self.strategy = MomentumStrategy(self.config)

        # Create sample market data
        self.sample_data = self._create_sample_market_data()

    def _create_sample_market_data(self):
        """Create sample market data for testing."""
        # Create a DataFrame with sample OHLCV data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='h')

        # Generate some dummy price data
        np.random.seed(42)  # For reproducibility
        close = 10000 + np.random.randn(100).cumsum() * 100

        # Add some patterns for testing signals
        # Bullish pattern near the end
        close[-10:] = 10000 + np.linspace(0, 500, 10)

        # Create the full OHLCV dataset
        high = close * (1 + np.random.rand(100) * 0.02)
        low = close * (1 - np.random.rand(100) * 0.02)
        open_price = low + (high - low) * np.random.rand(100)
        volume = (1000000 + np.random.randn(100).cumsum() * 100000).clip(min=100000)

        # Increase volume for the last few periods
        volume[-5:] = volume[-5:] * 2

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "ohlcv_data": df
        }

    def _create_bullish_market_data(self):
        """Create sample bullish market data for testing."""
        # Create a DataFrame with sample OHLCV data showing bullish pattern
        dates = pd.date_range(start='2020-01-01', periods=100, freq='h')

        # Generate bullish price data
        close = 10000 + np.cumsum(np.random.choice([10, 20, 30, -5, -10], size=100))

        # Make the end clearly bullish
        close[-20:] = np.linspace(close[-20], close[-20] * 1.2, 20)

        # Create the full OHLCV dataset
        high = close * (1 + np.random.rand(100) * 0.01)
        low = close * (1 - np.random.rand(100) * 0.01)
        open_prices = close - np.random.choice([-10, 10, 20], size=100)
        volume = np.random.randint(1000000, 5000000, size=100)

        # Increase volume at the end for confirmation
        volume[-5:] = volume[-5:] * 2

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        # Pre-calculate some indicators for easier testing
        df = IndicatorService.calculate_rsi(df, period=self.config["rsi_period"])
        df = IndicatorService.calculate_macd(
            df,
            fast=self.config["macd_fast"],
            slow=self.config["macd_slow"],
            signal=self.config["macd_signal"]
        )
        df = IndicatorService.calculate_ema(df, period=self.config["trend_ema_period"])
        df = IndicatorService.calculate_atr(df, period=14)

        # Ensure timestamp is the index
        if df.index.name != 'timestamp' and 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # Make the latest 3 data points have RSI in oversold territory for buy signals
        df.iloc[-3:, df.columns.get_loc('rsi')] = 25

        # Setup MACD bullish crossover
        df.iloc[-3, df.columns.get_loc('MACD_12_26_9')] = -0.5
        df.iloc[-2, df.columns.get_loc('MACD_12_26_9')] = 0
        df.iloc[-1, df.columns.get_loc('MACD_12_26_9')] = 0.5

        df.iloc[-3, df.columns.get_loc('MACDs_12_26_9')] = 0
        df.iloc[-2, df.columns.get_loc('MACDs_12_26_9')] = 0
        df.iloc[-1, df.columns.get_loc('MACDs_12_26_9')] = 0

        df.iloc[-3, df.columns.get_loc('MACDh_12_26_9')] = -0.5
        df.iloc[-2, df.columns.get_loc('MACDh_12_26_9')] = 0
        df.iloc[-1, df.columns.get_loc('MACDh_12_26_9')] = 3.0

        # Make prices clearly above EMA for trend alignment
        latest_ema = df.iloc[-1, df.columns.get_loc('ema_50')]
        # Cast to the same dtype as the close column to avoid FutureWarning
        df.iloc[-3:, df.columns.get_loc('close')] = df['close'].dtype.type(latest_ema * 1.05)

        # Increase volume dramatically for volume confirmation
        last_volume = df.iloc[-2, df.columns.get_loc('volume')]
        df.iloc[-1, df.columns.get_loc('volume')] = last_volume * 3

        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "ohlcv_data": df
        }

    def _create_bearish_market_data(self):
        """Create sample bearish market data for testing."""
        # Create a DataFrame with sample OHLCV data showing bearish pattern
        dates = pd.date_range(start='2020-01-01', periods=100, freq='h')

        # Generate bearish price data
        close = 10000 - np.cumsum(np.random.choice([10, 20, 30, -5, -10], size=100))

        # Make the end clearly bearish
        close[-20:] = np.linspace(close[-20], close[-20] * 0.8, 20)

        # Create the full OHLCV dataset
        high = close * (1 + np.random.rand(100) * 0.01)
        low = close * (1 - np.random.rand(100) * 0.01)
        open_prices = close + np.random.choice([-10, 10, 20], size=100)
        volume = np.random.randint(1000000, 5000000, size=100)

        # Increase volume at the end for confirmation
        volume[-5:] = volume[-5:] * 2

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        # Pre-calculate some indicators for easier testing
        df = IndicatorService.calculate_rsi(df, period=self.config["rsi_period"])
        df = IndicatorService.calculate_macd(
            df,
            fast=self.config["macd_fast"],
            slow=self.config["macd_slow"],
            signal=self.config["macd_signal"]
        )
        df = IndicatorService.calculate_ema(df, period=self.config["trend_ema_period"])
        df = IndicatorService.calculate_atr(df, period=14)

        # Ensure timestamp is the index
        if df.index.name != 'timestamp' and 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # Make the latest 3 data points have RSI in overbought territory for sell signals
        df.iloc[-3:, df.columns.get_loc('rsi')] = 75

        # Setup MACD bearish crossover
        df.iloc[-3, df.columns.get_loc('MACD_12_26_9')] = 0.5
        df.iloc[-2, df.columns.get_loc('MACD_12_26_9')] = 0
        df.iloc[-1, df.columns.get_loc('MACD_12_26_9')] = -0.5

        df.iloc[-3, df.columns.get_loc('MACDs_12_26_9')] = 0
        df.iloc[-2, df.columns.get_loc('MACDs_12_26_9')] = 0
        df.iloc[-1, df.columns.get_loc('MACDs_12_26_9')] = 0

        df.iloc[-3, df.columns.get_loc('MACDh_12_26_9')] = 0.5
        df.iloc[-2, df.columns.get_loc('MACDh_12_26_9')] = 0
        df.iloc[-1, df.columns.get_loc('MACDh_12_26_9')] = -3.0

        # Make prices clearly below EMA for trend alignment
        latest_ema = df.iloc[-1, df.columns.get_loc('ema_50')]
        # Cast to the same dtype as the close column to avoid FutureWarning
        df.iloc[-3:, df.columns.get_loc('close')] = df['close'].dtype.type(latest_ema * 0.95)

        # Increase volume dramatically for volume confirmation
        last_volume = df.iloc[-2, df.columns.get_loc('volume')]
        df.iloc[-1, df.columns.get_loc('volume')] = last_volume * 3

        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "ohlcv_data": df
        }

    def test_initialization(self):
        """Test that the strategy initializes correctly with given configuration."""
        strategy = MomentumStrategy(self.config)
        self.assertEqual(strategy.name, "MomentumStrategy")
        self.assertIsNotNone(strategy.description)
        self.assertEqual(strategy.config["rsi_period"], 14)
        self.assertEqual(strategy.config["rsi_oversold"], 30)
        self.assertEqual(strategy.config["rsi_overbought"], 70)

    def test_validate_config(self):
        """Test configuration validation."""
        # Test with invalid RSI levels
        invalid_config = self.config.copy()
        invalid_config["rsi_oversold"] = 40
        invalid_config["rsi_overbought"] = 30

        with self.assertRaises(ValueError):
            MomentumStrategy(invalid_config)

        # Test with invalid MACD parameters
        invalid_config = self.config.copy()
        invalid_config["macd_fast"] = 26
        invalid_config["macd_slow"] = 12

        with self.assertRaises(ValueError):
            MomentumStrategy(invalid_config)

        # Test with missing required parameter
        invalid_config = self.config.copy()
        del invalid_config["min_confidence_threshold"]

        with self.assertRaises(ValueError):
            MomentumStrategy(invalid_config)

    def test_generate_signals(self):
        """Test signal generation with different market data."""
        # Test with bullish market data
        bullish_data = self._create_bullish_market_data()

        # Temporarily lower the confidence threshold for testing
        original_threshold = self.strategy.config["min_confidence_threshold"]
        self.strategy.config["min_confidence_threshold"] = 0.3

        signals = self.strategy.generate_signals(bullish_data)

        # Restore original threshold
        self.strategy.config["min_confidence_threshold"] = original_threshold

        # We should have at least one buy signal in bullish data
        buy_signals = [s for s in signals if s["type"] == SignalType.BUY]
        self.assertTrue(len(buy_signals) > 0, "Should generate buy signals in bullish market")

        # Test with bearish market data
        bearish_data = self._create_bearish_market_data()

        # Temporarily lower the confidence threshold for testing
        self.strategy.config["min_confidence_threshold"] = 0.3

        signals = self.strategy.generate_signals(bearish_data)

        # Restore original threshold
        self.strategy.config["min_confidence_threshold"] = original_threshold

        # We should have at least one sell signal in bearish data
        sell_signals = [s for s in signals if s["type"] == SignalType.SELL]
        self.assertTrue(len(sell_signals) > 0, "Should generate sell signals in bearish market")

    def test_should_enter_position(self):
        """Test position entry logic."""
        # With bullish market data
        bullish_data = self._create_bullish_market_data()

        # Temporarily lower the confidence threshold for testing
        original_threshold = self.strategy.config["min_confidence_threshold"]
        self.strategy.config["min_confidence_threshold"] = 0.3

        should_enter, info = self.strategy.should_enter_position("BTC/USDT", bullish_data)

        # Restore original threshold
        self.strategy.config["min_confidence_threshold"] = original_threshold

        self.assertTrue(should_enter, "Should recommend position entry in bullish market")
        self.assertIn("signal", info, "Info should contain signal details")
        self.assertEqual(info["signal"]["type"], SignalType.BUY, "Signal type should be BUY")

        # With bearish market data
        bearish_data = self._create_bearish_market_data()
        should_enter, info = self.strategy.should_enter_position("BTC/USDT", bearish_data)

        # Shouldn't recommend entering a long position in a bearish market
        # (unless there's a specific buy signal which could happen due to test data)
        if should_enter:
            self.assertEqual(info["signal"]["type"], SignalType.BUY, "If entry recommended, signal should be BUY")

    def test_should_exit_position(self):
        """Test position exit logic."""
        # Create a mock position
        position = MagicMock(spec=Position)
        position.symbol = "BTC/USDT"
        position.position_type = PositionType.LONG
        position.should_exit.return_value = False

        # With bearish market data when we have a long position
        bearish_data = self._create_bearish_market_data()

        # Temporarily lower the confidence threshold for testing
        original_threshold = self.strategy.config["min_confidence_threshold"]
        self.strategy.config["min_confidence_threshold"] = 0.3

        should_exit, info = self.strategy.should_exit_position(position, bearish_data)

        # Restore original threshold
        self.strategy.config["min_confidence_threshold"] = original_threshold

        self.assertTrue(should_exit, "Should recommend exiting long position in bearish market")
        self.assertIn("signal", info, "Info should contain signal details")
        self.assertEqual(info["signal"]["type"], SignalType.SELL, "Signal type should be SELL for long exit")

        # With bullish market data when we have a short position
        position.position_type = PositionType.SHORT
        bullish_data = self._create_bullish_market_data()

        # Temporarily lower the confidence threshold for testing
        self.strategy.config["min_confidence_threshold"] = 0.3

        should_exit, info = self.strategy.should_exit_position(position, bullish_data)

        # Restore original threshold
        self.strategy.config["min_confidence_threshold"] = original_threshold

        self.assertTrue(should_exit, "Should recommend exiting short position in bullish market")
        self.assertIn("signal", info, "Info should contain signal details")
        self.assertEqual(info["signal"]["type"], SignalType.BUY, "Signal type should be BUY for short exit")

        # Test stop loss scenario
        position = MagicMock(spec=Position)
        position.symbol = "BTC/USDT"
        position.position_type = PositionType.LONG
        position.should_exit.return_value = True  # Stop loss hit

        neutral_data = self.sample_data  # Just use sample data
        should_exit, info = self.strategy.should_exit_position(position, neutral_data)

        self.assertTrue(should_exit, "Should exit when stop loss is hit")
        self.assertEqual(info["reason"], "Stop loss or take profit level reached")

    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Test with different account balances
        symbol = "BTC/USDT"

        # Small account balance
        small_balance = Decimal("1000")
        small_size = self.strategy.calculate_position_size(symbol, small_balance)

        # Medium account balance
        medium_balance = Decimal("10000")
        medium_size = self.strategy.calculate_position_size(symbol, medium_balance)

        # Large account balance
        large_balance = Decimal("100000")
        large_size = self.strategy.calculate_position_size(symbol, large_balance)

        # Check that position sizes scale with account balance
        self.assertLess(small_size, medium_size)
        self.assertLess(medium_size, large_size)

        # Check that position sizes are reasonable (not too large or small)
        self.assertLessEqual(small_size, small_balance)
        self.assertLessEqual(medium_size, medium_balance)
        self.assertLessEqual(large_size, large_balance)

        # Test with max_open_positions constraint
        restricted_config = self.config.copy()
        restricted_config["max_open_positions"] = 2
        restricted_strategy = MomentumStrategy(restricted_config)

        restricted_size = restricted_strategy.calculate_position_size(symbol, medium_balance)
        normal_size = self.strategy.calculate_position_size(symbol, medium_balance)

        # With more restrictions, position size should be smaller or equal
        self.assertLessEqual(restricted_size, normal_size)

    def test_stop_loss_calculation(self):
        """Test stop loss calculation."""
        symbol = "BTC/USDT"
        entry_price = Decimal("10000")

        # Add ATR to our sample data for testing
        market_data = self.sample_data.copy()
        market_data["ohlcv_data"] = IndicatorService.calculate_atr(market_data["ohlcv_data"])

        # Create BUY signals
        with patch.object(self.strategy, 'generate_signals') as mock_signals:
            mock_signals.return_value = [{"symbol": symbol, "type": SignalType.BUY}]

            # For a BUY signal (long position), stop loss should be below entry price
            stop_loss = self.strategy.get_stop_loss(symbol, entry_price, market_data)
            self.assertLess(stop_loss, entry_price, "Stop loss for long position should be below entry price")

        # Create SELL signals
        with patch.object(self.strategy, 'generate_signals') as mock_signals:
            mock_signals.return_value = [{"symbol": symbol, "type": SignalType.SELL}]

            # For a SELL signal (short position), stop loss should be above entry price
            stop_loss = self.strategy.get_stop_loss(symbol, entry_price, market_data)
            self.assertGreater(stop_loss, entry_price, "Stop loss for short position should be above entry price")

        # Test with different ATR multiplier
        custom_config = self.config.copy()
        custom_config["atr_multiplier"] = 1.0  # Tighter stop
        tighter_strategy = MomentumStrategy(custom_config)

        with patch.object(tighter_strategy, 'generate_signals') as mock_signals:
            mock_signals.return_value = [{"symbol": symbol, "type": SignalType.BUY}]

            regular_stop_loss = self.strategy.get_stop_loss(symbol, entry_price, market_data)
            tighter_stop_loss = tighter_strategy.get_stop_loss(symbol, entry_price, market_data)

            # Tighter stop loss should be closer to entry price
            self.assertGreater(tighter_stop_loss, regular_stop_loss,
                               "Tighter stop loss should be closer to entry price")

    def test_take_profit_calculation(self):
        """Test take profit calculation."""
        symbol = "BTC/USDT"
        entry_price = Decimal("10000")

        # Add ATR to our sample data for testing
        market_data = self.sample_data.copy()
        market_data["ohlcv_data"] = IndicatorService.calculate_atr(market_data["ohlcv_data"])

        # For a BUY signal (long position)
        with patch.object(self.strategy, 'generate_signals') as mock_signals:
            mock_signals.return_value = [{"symbol": symbol, "type": SignalType.BUY}]

            take_profits = self.strategy.get_take_profit(symbol, entry_price, market_data)

            # Should return multiple take profit levels
            self.assertIsInstance(take_profits, list, "Should return a list of take profit levels")
            self.assertEqual(len(take_profits), len(self.config["risk_reward_targets"]),
                             "Should have same number of take profits as risk_reward_targets")

            # For a long position, take profits should be above entry price
            for tp in take_profits:
                self.assertGreater(tp, entry_price, "Take profit for long position should be above entry price")

            # Take profits should be in ascending order
            self.assertEqual(take_profits, sorted(take_profits), "Take profits should be in ascending order")

        # For a SELL signal (short position)
        with patch.object(self.strategy, 'generate_signals') as mock_signals:
            mock_signals.return_value = [{"symbol": symbol, "type": SignalType.SELL}]

            take_profits = self.strategy.get_take_profit(symbol, entry_price, market_data)

            # For a short position, take profits should be below entry price
            for tp in take_profits:
                self.assertLess(tp, entry_price, "Take profit for short position should be below entry price")

            # Take profits should be in descending order
            self.assertEqual(take_profits, sorted(take_profits, reverse=True),
                             "Take profits for short position should be in descending order")

    def test_different_market_scenarios(self):
        """Test the strategy with different market scenarios."""
        # Test with sideways market
        dates = pd.date_range(start='2020-01-01', periods=100, freq='h')
        sideways_prices = 10000 + np.sin(np.linspace(0, 8*np.pi, 100)) * 200

        sideways_df = pd.DataFrame({
            'timestamp': dates,
            'open': sideways_prices,
            'high': sideways_prices * 1.01,
            'low': sideways_prices * 0.99,
            'close': sideways_prices,
            'volume': 1000000 + np.random.rand(100) * 500000
        })

        # Set the index to timestamp
        sideways_df = sideways_df.set_index('timestamp')

        # Calculate indicators
        sideways_df = IndicatorService.calculate_rsi(sideways_df, period=self.config["rsi_period"])
        sideways_df = IndicatorService.calculate_macd(
            sideways_df,
            fast=self.config["macd_fast"],
            slow=self.config["macd_slow"],
            signal=self.config["macd_signal"]
        )
        sideways_df = IndicatorService.calculate_ema(sideways_df, period=self.config["trend_ema_period"])

        sideways_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "ohlcv_data": sideways_df
        }

        signals = self.strategy.generate_signals(sideways_data)

        # In a sideways market, we should have both buy and sell signals or none
        buy_signals = [s for s in signals if s["type"] == SignalType.BUY]
        sell_signals = [s for s in signals if s["type"] == SignalType.SELL]

        self.assertTrue(
            (len(buy_signals) > 0 and len(sell_signals) > 0) or (len(buy_signals) == 0 and len(sell_signals) == 0),
            "In sideways market, should have both buy and sell signals or no signals at all"
        )

        # Test with extremely volatile market
        volatile_prices = 10000 + np.cumsum(np.random.normal(0, 300, 100))

        volatile_df = pd.DataFrame({
            'timestamp': dates,
            'open': volatile_prices,
            'high': volatile_prices * 1.05,
            'low': volatile_prices * 0.95,
            'close': volatile_prices,
            'volume': 1000000 + np.random.rand(100) * 1000000
        })

        # Set the index to timestamp
        volatile_df = volatile_df.set_index('timestamp')

        # Calculate indicators
        volatile_df = IndicatorService.calculate_rsi(volatile_df, period=self.config["rsi_period"])
        volatile_df = IndicatorService.calculate_macd(
            volatile_df,
            fast=self.config["macd_fast"],
            slow=self.config["macd_slow"],
            signal=self.config["macd_signal"]
        )
        volatile_df = IndicatorService.calculate_ema(volatile_df, period=self.config["trend_ema_period"])

        volatile_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "ohlcv_data": volatile_df
        }

        signals = self.strategy.generate_signals(volatile_data)

        # In a volatile market, signals should have confidence scores properly calculated
        for signal in signals:
            self.assertTrue(
                0 <= signal["confidence"] <= 1,
                f"Signal confidence should be between 0 and 1, got {signal['confidence']}"
            )

    def test_different_config_parameters(self):
        """Test the strategy with different configuration parameters."""
        # Create a more aggressive configuration
        aggressive_config = self.config.copy()
        aggressive_config["rsi_oversold"] = 40  # Less oversold (trigger buy signals earlier)
        aggressive_config["rsi_overbought"] = 60  # Less overbought (trigger sell signals earlier)
        aggressive_config["risk_per_trade"] = 3.0  # Higher risk per trade
        aggressive_config["atr_multiplier"] = 1.0  # Tighter stops

        aggressive_strategy = MomentumStrategy(aggressive_config)

        # Create a conservative configuration
        conservative_config = self.config.copy()
        conservative_config["rsi_oversold"] = 20  # More oversold (wait for deeper corrections)
        conservative_config["rsi_overbought"] = 80  # More overbought (let profits run longer)
        conservative_config["risk_per_trade"] = 1.0  # Lower risk per trade
        conservative_config["atr_multiplier"] = 3.0  # Wider stops

        conservative_strategy = MomentumStrategy(conservative_config)

        # Use the same market data for both
        bullish_data = self._create_bullish_market_data()

        # Aggressive strategy should generate more signals
        aggressive_signals = aggressive_strategy.generate_signals(bullish_data)
        conservative_signals = conservative_strategy.generate_signals(bullish_data)

        self.assertGreaterEqual(
            len(aggressive_signals), len(conservative_signals),
            "Aggressive strategy should generate more or equal signals than conservative"
        )

        # Compare position sizing
        symbol = "BTC/USDT"
        balance = Decimal("10000")

        aggressive_size = aggressive_strategy.calculate_position_size(symbol, balance)
        conservative_size = conservative_strategy.calculate_position_size(symbol, balance)

        self.assertGreater(
            aggressive_size, conservative_size,
            "Aggressive strategy should use larger position sizes"
        )

        # Compare stop loss distances
        entry_price = Decimal("10000")
        market_data = self.sample_data.copy()
        market_data["ohlcv_data"] = IndicatorService.calculate_atr(market_data["ohlcv_data"])

        with patch.object(aggressive_strategy, 'generate_signals') as mock_signals_agg, \
             patch.object(conservative_strategy, 'generate_signals') as mock_signals_cons:

            mock_signals_agg.return_value = [{"symbol": symbol, "type": SignalType.BUY}]
            mock_signals_cons.return_value = [{"symbol": symbol, "type": SignalType.BUY}]

            aggressive_stop = aggressive_strategy.get_stop_loss(symbol, entry_price, market_data)
            conservative_stop = conservative_strategy.get_stop_loss(symbol, entry_price, market_data)

            # Aggressive strategy should have tighter stops (closer to entry price)
            self.assertGreater(
                aggressive_stop, conservative_stop,
                "Aggressive strategy should have tighter stops (closer to entry price)"
            )


if __name__ == '__main__':
    unittest.main()
