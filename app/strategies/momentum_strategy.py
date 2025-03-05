"""
Momentum Trading Strategy Implementation.

This module implements a momentum-based trading strategy that combines
multiple technical indicators to generate trading signals.
"""

from typing import Dict, List, Tuple, Union, Any
from decimal import Decimal
import pandas as pd

from app.models.position import Position
from app.models.signals import SignalType
from app.strategies.base_strategy import BaseStrategy
from app.services.indicator_service import IndicatorService


class MomentumStrategy(BaseStrategy):
    """
    Momentum Trading Strategy.

    This strategy identifies momentum-based trading opportunities using a combination of
    indicators including RSI, MACD, volume analysis, and trend alignment. It generates
    buy signals when momentum is building up and sell signals when momentum is exhausting.

    Attributes:
        name: Name of the strategy
        description: Description of the strategy
        config: Configuration parameters
        logger: Logger instance for this strategy
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the momentum strategy with configuration parameters."""
        super().__init__(config)
        self.name = "MomentumStrategy"
        self.description = "Momentum-based trading strategy using RSI, MACD, and volume analysis"

        # Configure indicator service
        self.indicator_service = IndicatorService()

        # Validate momentum strategy specific config
        self._validate_momentum_config()

    def _validate_momentum_config(self) -> None:
        """Validate momentum strategy specific configuration."""
        required_params = [
            "rsi_period",
            "rsi_oversold",
            "rsi_overbought",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "volume_change_threshold",
            "trend_ema_period",
            "min_confidence_threshold",
        ]

        for param in required_params:
            if param not in self.config:
                raise ValueError(
                    f"Required parameter '{param}' not found in momentum strategy configuration"
                )

        # Validate RSI parameters
        if not (0 < self.config["rsi_oversold"] < self.config["rsi_overbought"] < 100):
            raise ValueError("RSI levels must satisfy: 0 < oversold < overbought < 100")

        # Validate MACD parameters
        if not (self.config["macd_fast"] < self.config["macd_slow"]):
            raise ValueError("MACD fast period must be less than slow period")

        # Validate volume threshold
        if self.config["volume_change_threshold"] <= 0:
            raise ValueError("Volume change threshold must be positive")

        # Validate confidence threshold
        if not (0 < self.config["min_confidence_threshold"] <= 1):
            raise ValueError("Minimum confidence threshold must be between 0 and 1")

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate buy/sell signals based on market data.

        Args:
            market_data: Dictionary containing market data for analysis with structure:
                {
                    "symbol": str,
                    "timeframe": str,
                    "ohlcv_data": pd.DataFrame,
                    ...
                }

        Returns:
            A list of signal dictionaries containing signal details
        """
        self.logger.info(
            f"Generating signals for {market_data['symbol']} on {market_data['timeframe']} timeframe"
        )

        signals = []
        ohlcv_data = market_data["ohlcv_data"]

        # Skip if not enough data
        if len(ohlcv_data) < max(
            self.config["rsi_period"],
            self.config["macd_slow"] + self.config["macd_signal"],
            self.config["trend_ema_period"],
        ):
            self.logger.warning(f"Not enough data for {market_data['symbol']}")
            return signals

        # Calculate indicators
        df = self._prepare_indicators(ohlcv_data)

        # Look for buy and sell signals in the most recent data
        latest_data = df.iloc[-5:].copy()  # Check last few candles

        for i in range(len(latest_data) - 1, -1, -1):
            row = latest_data.iloc[i]
            current_price = Decimal(str(row["close"]))

            # Check buy conditions
            buy_conditions = {
                "rsi": self._check_rsi_condition(row, "buy"),
                "macd": self._check_macd_condition(row, "buy"),
                "volume": self._check_volume_confirmation(latest_data, i),
                "trend": self._check_trend_alignment(row, "buy"),
            }

            # Check sell conditions
            sell_conditions = {
                "rsi": self._check_rsi_condition(row, "sell"),
                "macd": self._check_macd_condition(row, "sell"),
                "volume": self._check_volume_confirmation(latest_data, i),
                "trend": self._check_trend_alignment(row, "sell"),
            }

            # Calculate confidence scores
            buy_confidence = self._calculate_signal_confidence(buy_conditions)
            sell_confidence = self._calculate_signal_confidence(sell_conditions)

            # Generate signals if confidence exceeds threshold
            min_confidence = self.config["min_confidence_threshold"]

            if buy_confidence >= min_confidence and buy_confidence > sell_confidence:
                # Create buy signal
                signal = {
                    "symbol": market_data["symbol"],
                    "type": SignalType.BUY,
                    "price": current_price,
                    "confidence": buy_confidence,
                    "source": self.name,
                    "timeframe": market_data["timeframe"],
                    "indicators": {
                        "rsi": row["rsi"],
                        "macd": row["MACD_12_26_9"],
                        "macd_signal": row["MACDs_12_26_9"],
                        "macd_histogram": row["MACDh_12_26_9"],
                        "ema": row["ema_50"],
                    },
                    "metadata": {"conditions": buy_conditions},
                }
                signals.append(signal)
                self.logger.info(
                    f"Buy signal generated for {market_data['symbol']} with confidence {buy_confidence:.2f}"
                )

            elif sell_confidence >= min_confidence and sell_confidence > buy_confidence:
                # Create sell signal
                signal = {
                    "symbol": market_data["symbol"],
                    "type": SignalType.SELL,
                    "price": current_price,
                    "confidence": sell_confidence,
                    "source": self.name,
                    "timeframe": market_data["timeframe"],
                    "indicators": {
                        "rsi": row["rsi"],
                        "macd": row["MACD_12_26_9"],
                        "macd_signal": row["MACDs_12_26_9"],
                        "macd_histogram": row["MACDh_12_26_9"],
                        "ema": row["ema_50"],
                    },
                    "metadata": {"conditions": sell_conditions},
                }
                signals.append(signal)
                self.logger.info(
                    f"Sell signal generated for {market_data['symbol']} with confidence {sell_confidence:.2f}"
                )

        # Apply additional filters to the signals
        filtered_signals = [self._apply_filters(signal, market_data) for signal in signals]
        return [s for s in filtered_signals if s is not None]

    def _prepare_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators for the strategy."""
        df = ohlcv_data.copy()

        # Calculate RSI
        df = IndicatorService.calculate_rsi(df, period=self.config["rsi_period"])

        # Calculate MACD
        df = IndicatorService.calculate_macd(
            df,
            fast=self.config["macd_fast"],
            slow=self.config["macd_slow"],
            signal=self.config["macd_signal"],
        )

        # Calculate EMA for trend
        df = IndicatorService.calculate_ema(df, period=self.config["trend_ema_period"])

        # Calculate ATR for stop loss
        df = IndicatorService.calculate_atr(df, period=14)

        # Forward fill any NaN values that might be created
        df = df.ffill()

        return df

    def _check_rsi_condition(self, data: pd.Series, signal_type: str) -> bool:
        """
        Check RSI conditions for buy or sell signals.

        Args:
            data: Data row containing RSI value
            signal_type: Type of signal to check ('buy' or 'sell')

        Returns:
            Boolean indicating if RSI condition is met
        """
        if "rsi" not in data:
            return False

        # Extract scalar value to avoid Series comparison issues
        try:
            rsi_value = float(data["rsi"])
        except (ValueError, TypeError):
            return False

        if signal_type == "buy":
            # Buy when RSI crosses from below oversold to above oversold
            return (
                rsi_value < self.config["rsi_oversold"] + 5
                and rsi_value > self.config["rsi_oversold"]
            )
        else:
            # Sell when RSI crosses from above overbought to below overbought
            return (
                rsi_value > self.config["rsi_overbought"] - 5
                and rsi_value < self.config["rsi_overbought"]
            )

    def _check_macd_condition(self, data: pd.Series, signal_type: str) -> bool:
        """
        Check MACD conditions for buy or sell signals.

        Args:
            data: Data row containing MACD values
            signal_type: Type of signal to check ('buy' or 'sell')

        Returns:
            Boolean indicating if MACD condition is met
        """
        if "MACD_12_26_9" not in data or "MACDs_12_26_9" not in data or "MACDh_12_26_9" not in data:
            return False

        # Extract scalar values to avoid Series comparison issues
        try:
            macd_value = float(data["MACD_12_26_9"])
            signal_value = float(data["MACDs_12_26_9"])
            histogram_value = float(data["MACDh_12_26_9"])
        except (ValueError, TypeError):
            return False

        if signal_type == "buy":
            # Buy when MACD crosses above signal line or histogram turns positive
            return macd_value > signal_value and histogram_value > 0
        else:
            # Sell when MACD crosses below signal line or histogram turns negative
            return macd_value < signal_value and histogram_value < 0

    def _check_volume_confirmation(self, data: pd.DataFrame, index: int) -> bool:
        """
        Check volume conditions to confirm buy or sell signals.

        Args:
            data: DataFrame containing volume data
            index: Current row index

        Returns:
            Boolean indicating if volume condition is met
        """
        if index < 1 or "volume" not in data.columns:
            return False

        current_volume = data.iloc[index]["volume"]

        # Get average volume (last 5 periods)
        start_idx = max(0, index - 5)
        avg_volume = data.iloc[start_idx:index]["volume"].mean()

        # We want to see increased volume compared to the average
        return current_volume > avg_volume * self.config["volume_change_threshold"]

    def _check_trend_alignment(self, data: pd.Series, signal_type: str) -> bool:
        """
        Check if the overall trend aligns with the signal.

        Args:
            data: Data row containing price and EMA
            signal_type: Type of signal to check ('buy' or 'sell')

        Returns:
            Boolean indicating if the trend condition is met
        """
        if "close" not in data or "ema_50" not in data:
            return False

        # Extract scalar values to avoid Series comparison issues
        try:
            close_value = float(data["close"])
            ema_value = float(data["ema_50"])
        except (ValueError, TypeError):
            return False

        if signal_type == "buy":
            # For buy signals, price should be above EMA (uptrend)
            return close_value > ema_value
        else:
            # For sell signals, price should be below EMA (downtrend)
            return close_value < ema_value

    def _calculate_signal_confidence(self, conditions: Dict[str, bool]) -> float:
        """
        Calculate the confidence score of a signal based on conditions.

        Args:
            conditions: Dictionary of condition names and boolean results

        Returns:
            Confidence score between 0 and 1
        """
        # Define weights for each condition
        weights = {"rsi": 0.3, "macd": 0.3, "volume": 0.2, "trend": 0.2}

        # Calculate weighted sum
        confidence = sum(weights[key] if conditions[key] else 0 for key in conditions)

        return confidence

    def _apply_filters(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply additional filters to validate the signal.

        Args:
            signal: The signal dictionary
            market_data: The market data dictionary

        Returns:
            The filtered signal or None if filtered out
        """
        # Filter out signals that don't meet minimum criteria
        if signal["confidence"] < self.config["min_confidence_threshold"]:
            return None

        # We could add additional filters here
        # For example, checking against major support/resistance levels

        return signal

    def should_enter_position(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if a position should be entered for a given symbol.

        Args:
            symbol: The trading symbol to evaluate
            market_data: Dictionary containing market data for analysis

        Returns:
            A tuple containing:
                - A boolean indicating if a position should be entered
                - A dictionary with additional information about the decision
        """
        # Generate signals for the symbol
        signals = self.generate_signals(market_data)

        # Filter buy signals for this symbol
        buy_signals = [s for s in signals if s["symbol"] == symbol and s["type"] == SignalType.BUY]

        if not buy_signals:
            return False, {"reason": "No buy signals found"}

        # Get the strongest buy signal
        strongest_signal = max(buy_signals, key=lambda s: s["confidence"])

        # Check if the signal exceeds our confidence threshold
        should_enter = strongest_signal["confidence"] >= self.config["min_confidence_threshold"]

        info = {
            "signal": strongest_signal,
            "reason": (
                "Signal confidence meets threshold"
                if should_enter
                else "Signal confidence below threshold"
            ),
        }

        self.logger.info(
            f"Position entry decision for {symbol}: {should_enter}, confidence: {strongest_signal['confidence']:.2f}"
        )

        return should_enter, info

    def should_exit_position(
        self, position: Position, market_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if an existing position should be exited.

        Args:
            position: The current position to evaluate
            market_data: Dictionary containing market data for analysis

        Returns:
            A tuple containing:
                - A boolean indicating if the position should be exited
                - A dictionary with additional information about the decision
        """
        symbol = position.symbol
        signals = self.generate_signals(market_data)

        # Filter signals based on position type
        if position.position_type.value == "LONG":
            # For long positions, look for sell signals
            exit_signals = [
                s for s in signals if s["symbol"] == symbol and s["type"] == SignalType.SELL
            ]
        else:
            # For short positions, look for buy signals
            exit_signals = [
                s for s in signals if s["symbol"] == symbol and s["type"] == SignalType.BUY
            ]

        if not exit_signals:
            # Check if stop loss or take profit has been hit
            current_price = Decimal(str(market_data["ohlcv_data"].iloc[-1]["close"]))
            if position.should_exit(current_price):
                return True, {"reason": "Stop loss or take profit level reached"}

            return False, {"reason": "No exit signals found"}

        # Get the strongest exit signal
        strongest_signal = max(exit_signals, key=lambda s: s["confidence"])

        # Check if the signal exceeds our confidence threshold
        should_exit = strongest_signal["confidence"] >= self.config["min_confidence_threshold"]

        info = {
            "signal": strongest_signal,
            "reason": (
                "Signal confidence meets threshold"
                if should_exit
                else "Signal confidence below threshold"
            ),
        }

        self.logger.info(
            f"Position exit decision for {symbol}: {should_exit}, confidence: {strongest_signal['confidence']:.2f}"
        )

        return should_exit, info

    def calculate_position_size(self, symbol: str, account_balance: Decimal) -> Decimal:
        """Calculate the position size for a trade."""
        # Calculate risk amount based on risk percentage
        risk_amount = self.calculate_risk_amount(account_balance)

        # Calculate maximum position size based on max open positions
        max_position_size = account_balance / Decimal(self.config["max_open_positions"])

        # Use the smaller of the two calculated sizes to control risk
        position_size = min(risk_amount * Decimal(5), max_position_size)

        self.logger.info(f"Calculated position size for {symbol}: {position_size}")

        return position_size

    def get_stop_loss(
        self, symbol: str, entry_price: Decimal, market_data: Dict[str, Any]
    ) -> Decimal:
        """
        Determine the stop loss price for a trade.

        Args:
            symbol: The trading symbol
            entry_price: The entry price for the trade
            market_data: Dictionary containing market data for analysis

        Returns:
            The stop loss price
        """
        # Calculate ATR for dynamic stop loss
        ohlcv_data = market_data["ohlcv_data"]
        df = IndicatorService.calculate_atr(ohlcv_data)

        # Get the latest ATR value
        latest_atr = Decimal(str(df.iloc[-1]["atr"]))

        # Get the signal type (buy/sell)
        signals = self.generate_signals(market_data)
        signal_type = next(
            (s["type"] for s in signals if s["symbol"] == symbol),
            SignalType.BUY,  # Default to buy if no signal found
        )

        # Calculate stop loss based on ATR
        atr_multiplier = Decimal(str(self.config.get("atr_multiplier", 2.0)))

        if signal_type == SignalType.BUY:
            stop_loss = entry_price - (atr_multiplier * latest_atr)
        else:
            stop_loss = entry_price + (atr_multiplier * latest_atr)

        self.logger.info(f"Calculated stop loss for {symbol} at {stop_loss}, ATR: {latest_atr}")

        return stop_loss

    def get_take_profit(
        self, symbol: str, entry_price: Decimal, market_data: Dict[str, Any]
    ) -> Union[Decimal, List[Decimal]]:
        """
        Determine the take profit level(s) for a trade.

        Args:
            symbol: The trading symbol
            entry_price: The entry price for the trade
            market_data: Dictionary containing market data for analysis

        Returns:
            Either a single take profit price or a list of take profit prices
        """
        # Get the stop loss price
        stop_loss = self.get_stop_loss(symbol, entry_price, market_data)

        # Get the signal type (buy/sell)
        signals = self.generate_signals(market_data)
        signal_type = next(
            (s["type"] for s in signals if s["symbol"] == symbol),
            SignalType.BUY,  # Default to buy if no signal found
        )

        # Define risk-reward targets for multiple take profit levels
        if "risk_reward_targets" in self.config:
            risk_reward_targets = [
                Decimal(str(target)) for target in self.config["risk_reward_targets"]
            ]
        else:
            # Default targets if not configured
            risk_reward_targets = [Decimal("1.5"), Decimal("2.0"), Decimal("3.0")]

        # Calculate take profit levels based on risk/reward ratios
        risk = abs(entry_price - stop_loss)

        take_profit_levels = []
        for target in risk_reward_targets:
            reward = risk * target

            if signal_type == SignalType.BUY:
                take_profit = entry_price + reward
            else:
                take_profit = entry_price - reward

            take_profit_levels.append(take_profit)

        self.logger.info(f"Calculated take profit levels for {symbol}: {take_profit_levels}")

        return take_profit_levels

    def get_timeframes(self) -> List[str]:
        """Get the timeframes required by this strategy."""
        return self.config.get("timeframes", ["15m", "1h", "4h"])

    def get_required_indicators(self) -> List[str]:
        """Get the indicators required by this strategy."""
        return [
            f"rsi_{self.config['rsi_period']}",
            f"macd_{self.config['macd_fast']}_{self.config['macd_slow']}_{self.config['macd_signal']}",
            f"ema_{self.config['trend_ema_period']}",
            "atr_14",
        ]
