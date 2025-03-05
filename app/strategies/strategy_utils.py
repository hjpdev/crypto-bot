from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime


def calculate_risk_reward_ratio(
    entry: Decimal, stop_loss: Decimal, take_profit: Decimal
) -> Decimal:
    """
    Calculate the risk-to-reward ratio for a trade.

    Args:
        entry: The entry price for the trade
        stop_loss: The stop loss price for the trade
        take_profit: The take profit price for the trade

    Returns:
        The risk-to-reward ratio (reward/risk)

    Raises:
        ValueError: If prices are invalid or risk is zero
    """
    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        raise ValueError("Prices must be positive")

    # For long positions
    if entry > stop_loss and take_profit > entry:
        risk = entry - stop_loss
        reward = take_profit - entry
    # For short positions
    elif entry < stop_loss and take_profit < entry:
        risk = stop_loss - entry
        reward = entry - take_profit
    else:
        raise ValueError("Invalid price configuration for risk-reward calculation")

    if risk == 0:
        raise ValueError("Risk cannot be zero")

    return reward / risk


def validate_signal(signal: Dict, market_data: Dict, min_confidence: float = 0.7) -> bool:
    """
    Validate a trading signal against market data and minimum confidence level.

    Args:
        signal: Signal dictionary containing signal details
        market_data: Market data for validation
        min_confidence: Minimum confidence level required (0.0 to 1.0)

    Returns:
        Boolean indicating if the signal is valid
    """
    # Check if signal has required fields
    required_fields = ["symbol", "type", "confidence", "timestamp"]
    for field in required_fields:
        if field not in signal:
            return False

    # Check if confidence meets minimum threshold
    if signal["confidence"] < min_confidence:
        return False

    # Check if signal is recent (within last hour)
    signal_time = signal["timestamp"]
    if isinstance(signal_time, str):
        signal_time = datetime.fromisoformat(signal_time.replace("Z", "+00:00"))

    current_time = datetime.utcnow()
    if (current_time - signal_time).total_seconds() > 3600:  # 1 hour in seconds
        return False

    # Check if market data exists for the signal's symbol
    if signal["symbol"] not in market_data:
        return False

    return True


def calculate_signal_strength(indicators: Dict) -> float:
    """
    Calculate the strength of a signal based on multiple indicators.

    The function evaluates indicator values and returns a normalized strength score
    between 0.0 (weak) and 1.0 (strong).

    Args:
        indicators: Dictionary of indicator values and their parameters

    Returns:
        Signal strength as a float between 0.0 and 1.0
    """
    if not indicators:
        return 0.0

    strength_scores = []

    # Process trend indicators
    if "trend" in indicators:
        trend_indicators = indicators["trend"]

        # ADX (Average Directional Index)
        if "adx" in trend_indicators:
            adx_value = float(trend_indicators["adx"])
            # ADX > 25 indicates strong trend
            adx_score = min(1.0, max(0.0, (adx_value - 15) / 35))
            strength_scores.append(adx_score)

        # Moving averages
        if "ema_cross" in trend_indicators:
            ema_cross = trend_indicators["ema_cross"]
            if ema_cross == "bullish":
                strength_scores.append(0.8)
            elif ema_cross == "bearish":
                strength_scores.append(0.8)

    # Process momentum indicators
    if "momentum" in indicators:
        momentum_indicators = indicators["momentum"]

        # RSI (Relative Strength Index)
        if "rsi" in momentum_indicators:
            rsi_value = float(momentum_indicators["rsi"])
            # RSI near extremes (oversold/overbought)
            if rsi_value <= 30:  # Oversold (bullish)
                rsi_score = 1.0 - (rsi_value / 30)
            elif rsi_value >= 70:  # Overbought (bearish)
                rsi_score = (rsi_value - 70) / 30
            else:
                rsi_score = 0.5
            strength_scores.append(rsi_score)

        # MACD
        if "macd" in momentum_indicators:
            macd_signal = momentum_indicators["macd"]
            if macd_signal == "bullish_crossover" or macd_signal == "bearish_crossover":
                strength_scores.append(0.9)
            elif macd_signal == "bullish_divergence" or macd_signal == "bearish_divergence":
                strength_scores.append(0.7)

    # Process volatility indicators
    if "volatility" in indicators:
        volatility_indicators = indicators["volatility"]

        # Bollinger Bands
        if "bb_width" in volatility_indicators:
            bb_width = float(volatility_indicators["bb_width"])
            # Narrow bands (< 0.1) suggest potential breakout
            if bb_width < 0.1:
                strength_scores.append(0.8)

        # ATR (Average True Range)
        if "atr_percent" in volatility_indicators:
            atr_percent = float(volatility_indicators["atr_percent"])
            # High volatility can indicate strong moves
            atr_score = min(1.0, atr_percent / 5.0)
            strength_scores.append(atr_score)

    # If no scores were calculated, return 0
    if not strength_scores:
        return 0.0

    # Apply weights to different categories if provided
    # For now, simple average
    return sum(strength_scores) / len(strength_scores)


def combine_signal_sources(signals: List[Dict], weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Combine multiple signal sources with optional weighting.

    Args:
        signals: List of signal dictionaries from various sources
        weights: Optional dictionary mapping source names to weights

    Returns:
        Combined signal dictionary
    """
    if not signals:
        return {"type": "neutral", "confidence": 0.0, "sources": []}

    # Default weights if not provided
    if weights is None:
        weights = {signal["source"]: 1.0 for signal in signals}

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Track buy and sell sentiment
    buy_confidence = 0.0
    sell_confidence = 0.0
    used_sources = []

    for signal in signals:
        source = signal.get("source", "unknown")
        weight = normalized_weights.get(source, 0.1)  # Default weight if source not in weights

        if signal["type"].upper() == "BUY":
            buy_confidence += signal["confidence"] * weight
            used_sources.append(source)
        elif signal["type"].upper() == "SELL":
            sell_confidence += signal["confidence"] * weight
            used_sources.append(source)

    # Determine the combined signal type
    if buy_confidence > sell_confidence:
        signal_type = "BUY"
        confidence = buy_confidence - sell_confidence
    elif sell_confidence > buy_confidence:
        signal_type = "SELL"
        confidence = sell_confidence - buy_confidence
    else:
        signal_type = "NEUTRAL"
        confidence = 0.0

    # Cap confidence at 1.0
    confidence = min(1.0, confidence)

    return {
        "type": signal_type,
        "confidence": confidence,
        "sources": list(set(used_sources)),
        "raw_signals": signals,
    }


def calculate_dynamic_stop_loss(
    entry_price: Decimal, atr: Decimal, multiplier: float = 2.0
) -> Decimal:
    """Calculate a dynamic stop loss based on Average True Range."""
    atr_value = atr * Decimal(str(multiplier))

    # For long positions
    return entry_price - atr_value


def calculate_dynamic_take_profit(
    entry_price: Decimal, stop_loss: Decimal, risk_reward_targets: List[float] = [1.5, 2.5, 3.5]
) -> List[Decimal]:
    """
    Calculate dynamic take profit levels based on risk-reward ratios.

    Args:
        entry_price: Entry price for the position
        stop_loss: Stop loss price
        risk_reward_targets: List of risk-reward ratios for multiple take profit levels

    Returns:
        List of take profit prices
    """
    risk = abs(entry_price - stop_loss)

    # For long positions (assuming entry > stop)
    if entry_price > stop_loss:
        return [entry_price + (risk * Decimal(str(rr))) for rr in risk_reward_targets]
    # For short positions (assuming entry < stop)
    else:
        return [entry_price - (risk * Decimal(str(rr))) for rr in risk_reward_targets]


def is_confirmed_by_volume(
    signal_type: str,
    price_change: Decimal,
    volume_change: Decimal,
    min_volume_increase: float = 1.5,
) -> bool:
    """
    Check if a price move is confirmed by sufficient volume.

    Args:
        signal_type: Type of signal ("BUY" or "SELL")
        price_change: Percentage price change
        volume_change: Ratio of current volume to average volume
        min_volume_increase: Minimum volume increase required for confirmation

    Returns:
        Boolean indicating if the move is confirmed by volume
    """
    # Price increase with volume increase suggests strong buying pressure
    if signal_type.upper() == "BUY" and price_change > 0 and volume_change >= min_volume_increase:
        return True

    # Price decrease with volume increase suggests strong selling pressure
    if signal_type.upper() == "SELL" and price_change < 0 and volume_change >= min_volume_increase:
        return True

    return False
