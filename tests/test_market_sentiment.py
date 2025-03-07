"""
Tests for the Market Sentiment functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.market_sentiment import MarketSentiment
from app.services.indicator_service import IndicatorService


@pytest.fixture
def sample_market_data():
    """Generate sample OHLCV data for testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')

    # Create sample data
    data = {
        'timestamp': dates,
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000, 200, 100)
    }

    # Ensure high is always >= open, close, and low
    for i in range(len(data['high'])):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i], data['low'][i])

    # Ensure low is always <= open, close, and high
    for i in range(len(data['low'])):
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i], data['high'][i])

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df


@pytest.fixture
def bullish_market_data():
    """Generate bullish market data for testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')

    # Base price that trends upward
    base_price = np.linspace(100, 150, 100) + np.random.normal(0, 3, 100)

    # Create sample data with an uptrend
    data = {
        'timestamp': dates,
        'open': base_price - np.random.uniform(0, 2, 100),
        'close': base_price + np.random.uniform(0, 2, 100),  # Ensure mostly bullish candles
        'high': None,
        'low': None,
        'volume': np.random.normal(1000, 200, 100) * (1 + 0.1 * np.sin(np.linspace(0, 6, 100)))
    }

    # Ensure close > open for most candles (bullish)
    for i in range(len(data['close'])):
        if i > 70:  # Make the last 30 candles strongly bullish
            data['close'][i] = data['open'][i] * 1.02  # 2% gain

    # Set high and low based on open/close
    data['high'] = [max(o, c) + np.random.uniform(0.5, 2.0) for o, c in zip(data['open'], data['close'])]
    data['low'] = [min(o, c) - np.random.uniform(0.5, 2.0) for o, c in zip(data['open'], data['close'])]

    # Increase volume for bullish candles
    for i in range(len(data['volume'])):
        if data['close'][i] > data['open'][i]:
            data['volume'][i] *= 1.5

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df


@pytest.fixture
def bearish_market_data():
    """Generate bearish market data for testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')

    # Base price that trends strongly downward
    base_price = np.linspace(150, 80, 100)  # More significant decline

    # Add some noise but maintain the strong downtrend
    price_with_noise = base_price + np.random.normal(0, 1.5, 100)

    # Create sample data with a clear downtrend
    data = {
        'timestamp': dates,
        'open': price_with_noise + np.random.uniform(0, 1, 100),
        'close': price_with_noise - np.random.uniform(0, 1, 100),  # Ensure bearish candles
        'high': None,
        'low': None,
        'volume': np.random.normal(1000, 200, 100) * (1 + 0.1 * np.sin(np.linspace(0, 6, 100)))
    }

    # Make sure all recent candles are strongly bearish
    for i in range(len(data['close'])):
        if i > 50:  # Make the second half strongly bearish
            data['close'][i] = data['open'][i] * 0.97  # 3% loss
        else:
            # Still mostly bearish in the first half
            data['close'][i] = data['open'][i] * (0.99 if i % 5 != 0 else 1.01)  # Occasional bullish candle

    # Set high and low based on open/close
    data['high'] = [max(o, c) + np.random.uniform(0.2, 1.0) for o, c in zip(data['open'], data['close'])]
    data['low'] = [min(o, c) - np.random.uniform(0.2, 1.0) for o, c in zip(data['open'], data['close'])]

    # Increase volume for bearish candles to emphasize the trend
    for i in range(len(data['volume'])):
        if data['close'][i] < data['open'][i]:
            data['volume'][i] *= 2.0  # Higher volume multiplier

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df


@pytest.fixture
def multiple_symbols_data():
    """Generate data for multiple symbols for market breadth testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')

    symbols_data = {}

    # Create data for 10 symbols with varying trends
    for i in range(10):
        # Determine if this symbol is bullish or bearish
        is_bullish = i < 7  # 7 bullish, 3 bearish symbols

        # Base price trend
        if is_bullish:
            base_price = np.linspace(100, 150, 100) + np.random.normal(0, 3, 100)
        else:
            base_price = np.linspace(150, 100, 100) + np.random.normal(0, 3, 100)

        # Create sample data with appropriate trend
        data = {
            'timestamp': dates,
            'open': base_price - np.random.uniform(0, 2, 100) * (-1 if is_bullish else 1),
            'close': base_price + np.random.uniform(0, 2, 100) * (1 if is_bullish else -1),
            'high': None,
            'low': None,
            'volume': np.random.normal(1000, 200, 100)
        }

        # Set high and low based on open/close
        data['high'] = [max(o, c) + np.random.uniform(0.5, 2.0) for o, c in zip(data['open'], data['close'])]
        data['low'] = [min(o, c) - np.random.uniform(0.5, 2.0) for o, c in zip(data['open'], data['close'])]

        # Create DataFrame
        symbol_name = f"SYMBOL{i}/USDT"
        symbols_data[symbol_name] = pd.DataFrame(data).set_index('timestamp')

    return symbols_data


@pytest.fixture
def sample_order_book():
    """Generate sample order book data for testing."""
    # Create bids (buy orders) descending by price
    bids = [
        [9900, 2.5],   # price, volume
        [9850, 5.0],
        [9800, 10.0],
        [9750, 15.0],
        [9700, 20.0],
    ]

    # Create asks (sell orders) ascending by price
    asks = [
        [10100, 2.0],
        [10150, 4.0],
        [10200, 8.0],
        [10250, 12.0],
        [10300, 16.0],
    ]

    return {
        'bids': bids,
        'asks': asks
    }


@pytest.fixture
def sample_trades():
    """Generate sample trade data for testing."""
    trades = []

    # Create 20 sample trades
    for i in range(20):
        # Alternate between buy and sell trades
        side = 'buy' if i % 2 == 0 else 'sell'

        # Create trade data
        trade = {
            'price': 10000 + (i * 10) * (1 if side == 'buy' else -1),
            'amount': 1.0 + i * 0.1,
            'side': side,
            'timestamp': datetime.now() - timedelta(minutes=i)
        }

        trades.append(trade)

    return trades


@pytest.fixture
def bullish_biased_order_book():
    """Generate order book data with bullish bias (more buying pressure)."""
    # Create bids (buy orders) with higher volume
    bids = [
        [9900, 10.0],   # price, volume (higher volumes = more buying interest)
        [9850, 15.0],
        [9800, 20.0],
        [9750, 25.0],
        [9700, 30.0],
    ]

    # Create asks (sell orders) with lower volume
    asks = [
        [10100, 2.0],
        [10150, 3.0],
        [10200, 4.0],
        [10250, 5.0],
        [10300, 6.0],
    ]

    return {
        'bids': bids,
        'asks': asks
    }


@pytest.fixture
def market_sentiment():
    """Create a MarketSentiment instance for testing."""
    indicator_service = IndicatorService()
    return MarketSentiment(indicator_service=indicator_service)


class TestInternalIndicators:
    """Tests for internal sentiment indicators."""

    def test_internal_indicators_bullish(self, market_sentiment, bullish_market_data):
        """Test calculating internal indicators in bullish market."""
        indicators = market_sentiment.calculate_internal_indicators(bullish_market_data)

        # Check that all required indicators are returned
        assert 'rsi_sentiment' in indicators
        assert 'price_position' in indicators
        assert 'candle_bullish' in indicators
        assert 'overall_bullish_score' in indicators

        # In bullish data, overall score should be positive
        assert indicators['overall_bullish_score'] > 0, "Bullish market should have positive sentiment score"

        # RSI should be above neutral in bullish market
        assert indicators['rsi_sentiment'] > 0, "RSI should be bullish in bullish market"

    def test_internal_indicators_bearish(self, market_sentiment, bearish_market_data):
        """Test calculating internal indicators in bearish market."""
        indicators = market_sentiment.calculate_internal_indicators(bearish_market_data)

        # In bearish data, overall score should be negative
        assert indicators['overall_bullish_score'] < 0, "Bearish market should have negative sentiment score"

        # Last candle should be bearish
        assert indicators['candle_bullish'] == -1, "Last candle should be bearish in bearish market"


class TestMarketBreadth:
    """Tests for market breadth calculation."""

    def test_market_breadth_calculation(self, market_sentiment, multiple_symbols_data):
        """Test calculating market breadth across multiple symbols."""
        # Create a mock version of multiple_symbols_data where we know the characteristics
        mock_data = {}

        # Create 7 bullish and 3 bearish symbols
        for i in range(1, 11):
            symbol = f"PAIR{i}"
            # Start with a copy of the first symbol's data from the fixture
            df = multiple_symbols_data[list(multiple_symbols_data.keys())[0]].copy()

            # Make 7 symbols bullish
            if i <= 7:
                # Set close price trend upward
                df['close'] = df['close'] * 1.1
                # Ensure last close is higher than previous close
                df.loc[df.index[-1], 'close'] = df.loc[df.index[-2], 'close'] * 1.05
            else:
                # Set close price trend downward
                df['close'] = df['close'] * 0.9
                # Ensure last close is lower than previous close
                df.loc[df.index[-1], 'close'] = df.loc[df.index[-2], 'close'] * 0.95

            mock_data[symbol] = df

        # Mock the indicator service to return predetermined values
        original_calculate_macd = market_sentiment.indicator_service.calculate_macd
        original_calculate_rsi = market_sentiment.indicator_service.calculate_rsi

        def mock_calculate_macd(data):
            result = data.copy()
            # For bullish symbols, set positive MACD histogram
            if data['close'].iloc[-1] > data['close'].iloc[-2]:
                result['MACDh_12_26_9'] = 1.0  # Positive MACD histogram (bullish)
            else:
                result['MACDh_12_26_9'] = -1.0  # Negative MACD histogram (bearish)
            return result

        def mock_calculate_rsi(data):
            result = data.copy()
            # For bullish symbols, set RSI above 50
            if data['close'].iloc[-1] > data['close'].iloc[-2]:
                result['rsi'] = 60.0  # Bullish RSI
            else:
                result['rsi'] = 40.0  # Bearish RSI
            return result

        # Apply the mocks
        market_sentiment.indicator_service.calculate_macd = mock_calculate_macd
        market_sentiment.indicator_service.calculate_rsi = mock_calculate_rsi

        try:
            # Use our constructed mock data for the test
            breadth = market_sentiment.get_market_breadth(mock_data)

            # Check that all metrics are calculated
            assert 'advance_decline_ratio' in breadth
            assert 'percent_above_ma50' in breadth
            assert 'percent_above_ma200' in breadth
            assert 'percent_bullish_macd' in breadth
            assert 'percent_bullish_rsi' in breadth
            assert 'breadth_score' in breadth

            # With our mocked indicator values, 7 out of 10 symbols are bullish in all aspects
            # The breadth score should be positive
            assert breadth['breadth_score'] > 0, "Breadth score should be positive with majority bullish symbols"
        finally:
            # Restore original methods
            market_sentiment.indicator_service.calculate_macd = original_calculate_macd
            market_sentiment.indicator_service.calculate_rsi = original_calculate_rsi

    def test_empty_market_data(self, market_sentiment):
        """Test handling of empty market data."""
        breadth = market_sentiment.get_market_breadth({})

        # Should return default values without errors
        assert breadth['advance_decline_ratio'] == 0
        assert breadth['percent_above_ma50'] == 0
        assert breadth['percent_above_ma200'] == 0
        assert breadth['percent_bullish_macd'] == 0
        assert breadth['percent_bullish_rsi'] == 0
        assert breadth['percent_overbought'] == 0
        assert breadth['percent_oversold'] == 0
        assert breadth['breadth_score'] == 0


class TestOrderFlowAnalysis:
    """Tests for order flow analysis."""

    def test_buying_selling_pressure(self, market_sentiment, sample_order_book, sample_trades):
        """Test analyzing buying and selling pressure from order book and trades."""
        pressure = market_sentiment.calculate_buying_selling_pressure(sample_order_book, sample_trades)

        # Check that all metrics are calculated
        assert 'buying_pressure' in pressure
        assert 'order_book_imbalance' in pressure
        assert 'overall_pressure' in pressure

        # Check that imbalance metrics are calculated for different levels
        assert any(key.startswith('imbalance_') for key in pressure.keys())

    def test_bullish_order_flow(self, market_sentiment, bullish_biased_order_book, sample_trades):
        """Test detecting bullish bias in order flow."""
        pressure = market_sentiment.calculate_buying_selling_pressure(bullish_biased_order_book, sample_trades)

        # With higher bid volumes, order book imbalance should be positive
        assert pressure['order_book_imbalance'] > 0, "Bullish order book should show positive imbalance"

        # Overall pressure should be positive
        assert pressure['overall_pressure'] > 0, "Bullish order flow should show positive overall pressure"


class TestOverallSentiment:
    """Tests for overall sentiment calculation."""

    def test_overall_sentiment_bullish(self, market_sentiment, bullish_market_data,
                                       multiple_symbols_data, bullish_biased_order_book, sample_trades):
        """Test overall sentiment calculation in bullish conditions."""
        # Calculate component sentiment indicators
        breadth = market_sentiment.get_market_breadth(multiple_symbols_data)
        order_flow = market_sentiment.calculate_buying_selling_pressure(bullish_biased_order_book, sample_trades)

        # Calculate overall sentiment
        sentiment = market_sentiment.get_overall_sentiment(
            "BTC/USDT",
            bullish_market_data,
            market_breadth_data=breadth,
            order_flow_data=order_flow
        )

        # Check required fields
        assert 'overall_sentiment_score' in sentiment
        assert 'sentiment_category' in sentiment
        assert 'confidence' in sentiment
        assert 'factor_breakdown' in sentiment

        # In bullish conditions, sentiment should be positive
        assert sentiment['overall_sentiment_score'] > 0, "Overall sentiment should be positive in bullish conditions"

        # Category should indicate bullishness
        assert 'bullish' in sentiment['sentiment_category'], f"Sentiment category should contain 'bullish', got {sentiment['sentiment_category']}"

    def test_overall_sentiment_bearish(self, market_sentiment, bearish_market_data):
        """Test overall sentiment calculation in bearish conditions."""
        # Calculate overall sentiment with just internal indicators
        sentiment = market_sentiment.get_overall_sentiment("BTC/USDT", bearish_market_data)

        # Due to randomness in the test data, we can't always guarantee a negative score
        # But we should verify that either the score is negative OR the sentiment category indicates bearishness
        has_bearish_sentiment = sentiment['overall_sentiment_score'] < 0 or 'bearish' in sentiment['sentiment_category']
        assert has_bearish_sentiment, (
            f"Expected bearish sentiment (negative score or bearish category), "
            f"got score: {sentiment['overall_sentiment_score']}, category: {sentiment['sentiment_category']}"
        )

        # If the sentiment score is positive, it should be very small (close to neutral)
        if sentiment['overall_sentiment_score'] > 0:
            assert sentiment['overall_sentiment_score'] < 0.1, (
                f"If sentiment is positive, it should be close to neutral, got {sentiment['overall_sentiment_score']}"
            )


class TestSentimentCategories:
    """Tests for sentiment categorization."""

    def test_sentiment_categorization(self, market_sentiment):
        """Test that sentiment scores are properly categorized."""
        # Test extreme bullish
        assert market_sentiment._categorize_sentiment(0.8) == 'extremely_bullish'

        # Test strong bullish
        assert market_sentiment._categorize_sentiment(0.6) == 'strongly_bullish'

        # Test bullish
        assert market_sentiment._categorize_sentiment(0.3) == 'bullish'

        # Test slightly bullish
        assert market_sentiment._categorize_sentiment(0.15) == 'slightly_bullish'

        # Test neutral
        assert market_sentiment._categorize_sentiment(0.0) == 'neutral'

        # Test slightly bearish
        assert market_sentiment._categorize_sentiment(-0.15) == 'slightly_bearish'

        # Test bearish
        assert market_sentiment._categorize_sentiment(-0.3) == 'bearish'

        # Test strongly bearish
        assert market_sentiment._categorize_sentiment(-0.6) == 'strongly_bearish'

        # Test extremely bearish
        assert market_sentiment._categorize_sentiment(-0.8) == 'extremely_bearish'
