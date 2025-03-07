"""
Opportunity scanner service for identifying trading opportunities.

This module provides functionality for scanning markets and identifying
potential trading opportunities based on configured strategies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from app.core.database import Database
from app.core.exceptions import DataError, ExchangeError
from app.models.signals import Signal, SignalType
from app.services.exchange_service import ExchangeService
from app.services.market_analyzer import MarketAnalyzer
from app.services.data_preparation import ohlcv_to_dataframe


class Opportunity(BaseModel):
    """
    Represents a trading opportunity identified by the scanner.

    An opportunity contains information about a potential trade, including
    the symbol, entry price, confidence level, and other relevant metrics.
    """

    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signal_type: SignalType
    timeframe: str
    entry_price: float
    confidence: float = Field(ge=0.0, le=1.0)
    strategy_name: str
    indicators: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class OpportunityScanner:
    """
    Scans the market for trading opportunities.

    This class is responsible for periodically scanning markets and
    identifying potential trading opportunities based on configured
    strategies. It filters and prioritizes opportunities before
    recording them for further processing.
    """

    def __init__(
        self,
        exchange_service: ExchangeService,
        market_analyzer: MarketAnalyzer,
        database: Optional[Database] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the opportunity scanner.

        Args:
            exchange_service: Service for interacting with exchanges
            market_analyzer: Service for analyzing market data
            database: Database instance for storing opportunities
            logger: Logger instance
        """
        self.exchange_service = exchange_service
        self.market_analyzer = market_analyzer
        self.database = database
        self.logger = logger or logging.getLogger("services.scanner")
        self.strategies = []  # Will be set externally
        self.default_timeframes = ["1h", "4h", "1d"]  # Default timeframes to analyze

    def add_strategy(self, strategy: Any) -> None:
        """
        Add a strategy to use for scanning.

        Args:
            strategy: Strategy instance to add
        """
        self.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.__class__.__name__}")

    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy by name.

        Args:
            strategy_name: Name of the strategy to remove

        Returns:
            bool: True if strategy was removed, False if not found
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.__class__.__name__ == strategy_name:
                del self.strategies[i]
                self.logger.info(f"Removed strategy: {strategy_name}")
                return True

        self.logger.warning(f"Strategy not found: {strategy_name}")
        return False

    def scan_markets(
        self, symbols: List[str], timeframes: Optional[List[str]] = None
    ) -> List[Opportunity]:
        """
        Scan multiple markets for trading opportunities.

        Args:
            symbols: List of symbols to scan
            timeframes: List of timeframes to analyze, defaults to self.default_timeframes

        Returns:
            List of identified opportunities

        Raises:
            DataError: If market data cannot be retrieved
        """
        timeframes = timeframes or self.default_timeframes
        all_opportunities = []

        self.logger.info(f"Scanning {len(symbols)} markets across {len(timeframes)} timeframes")

        for symbol in symbols:
            try:
                # Get market data for the symbol
                market_data = self.get_required_market_data(symbol, timeframes)

                # Evaluate opportunities for this symbol
                symbol_opportunities = self.evaluate_opportunity(symbol, market_data)
                all_opportunities.extend(symbol_opportunities)

            except (ExchangeError, DataError) as e:
                self.logger.warning(f"Error scanning market {symbol}: {e}")
                continue
            except Exception as e:
                self.logger.exception(f"Unexpected error scanning market {symbol}: {e}")
                continue

        # Filter and prioritize opportunities
        filtered_opportunities = self.filter_opportunities(all_opportunities)
        prioritized_opportunities = self.prioritize_opportunities(filtered_opportunities)

        self.logger.info(
            f"Scan completed. Found {len(all_opportunities)} opportunities, "
            f"{len(filtered_opportunities)} after filtering, "
            f"{len(prioritized_opportunities)} prioritized."
        )

        # Record opportunities
        for opportunity in prioritized_opportunities:
            try:
                self.record_opportunity(opportunity)
            except Exception as e:
                self.logger.error(f"Error recording opportunity for {opportunity.symbol}: {e}")

        return prioritized_opportunities

    def evaluate_opportunity(
        self, symbol: str, market_data: Dict[str, pd.DataFrame]
    ) -> List[Opportunity]:
        """
        Evaluate a single market for trading opportunities.

        Args:
            symbol: Symbol to evaluate
            market_data: Dictionary of market data by timeframe

        Returns:
            List of identified opportunities
        """
        opportunities = []

        if not self.strategies:
            self.logger.warning("No strategies configured for evaluation")
            return []

        self.logger.debug(f"Evaluating {symbol} with {len(self.strategies)} strategies")

        # Apply each strategy to evaluate opportunities
        for strategy in self.strategies:
            try:
                strategy_name = strategy.__class__.__name__

                # Check each timeframe
                for timeframe, data in market_data.items():
                    if data.empty:
                        continue

                    # Analyze market conditions
                    regime = self.market_analyzer.detect_market_regime(data)

                    # Apply strategy to get signals
                    should_enter, entry_info = strategy.should_enter_position(
                        symbol,
                        {
                            "ohlcv_data": data,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "market_regime": regime,
                        },
                    )

                    if should_enter:
                        # Create opportunity from signal
                        opportunity = Opportunity(
                            symbol=symbol,
                            signal_type=(
                                SignalType.BUY
                                if entry_info.get("direction", "long") == "long"
                                else SignalType.SELL
                            ),
                            timeframe=timeframe,
                            entry_price=float(data["close"].iloc[-1]),
                            confidence=entry_info.get("confidence", 0.5),
                            strategy_name=strategy_name,
                            indicators=entry_info.get("indicators", {}),
                            metadata={
                                "market_regime": regime,
                                "reason": entry_info.get("reason", ""),
                                "strategy_specific": entry_info.get("metadata", {}),
                            },
                        )
                        opportunities.append(opportunity)
                        self.logger.debug(
                            f"Found opportunity for {symbol} on {timeframe}: {entry_info.get('reason', '')}"
                        )

            except Exception as e:
                self.logger.exception(
                    f"Error applying strategy {strategy.__class__.__name__} to {symbol}: {e}"
                )
                continue

        return opportunities

    def filter_opportunities(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """
        Apply additional filters to the opportunities.

        Args:
            opportunities: List of opportunities to filter

        Returns:
            Filtered list of opportunities
        """
        if not opportunities:
            return []

        filtered = []

        for opportunity in opportunities:
            # Filter out low confidence opportunities
            if opportunity.confidence < 0.3:
                continue

            # Additional filtering logic can be added here:
            # - Volume requirements
            # - Volatility checks
            # - Correlation with market benchmarks
            # - Recent performance

            filtered.append(opportunity)

        return filtered

    def prioritize_opportunities(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """
        Rank opportunities by quality/priority.

        Args:
            opportunities: List of opportunities to prioritize

        Returns:
            Prioritized list of opportunities
        """
        if not opportunities:
            return []

        # Calculate priority scores
        for opportunity in opportunities:
            priority_score = 0

            # Base score from confidence
            priority_score += int(opportunity.confidence * 100)

            # Higher priority for daily timeframe signals
            if opportunity.timeframe == "1d":
                priority_score += 50
            elif opportunity.timeframe == "4h":
                priority_score += 30
            elif opportunity.timeframe == "1h":
                priority_score += 10

            # Add other priority factors here:
            # - Historical strategy performance
            # - Current market conditions
            # - Asset volatility
            # - Trading volume

            opportunity.priority = priority_score

        # Sort by priority (descending)
        prioritized = sorted(opportunities, key=lambda x: x.priority, reverse=True)

        return prioritized

    def record_opportunity(self, opportunity: Opportunity) -> None:
        """
        Save an opportunity to the database.

        Args:
            opportunity: Opportunity to record

        Raises:
            DatabaseError: If opportunity cannot be saved
        """
        # Convert opportunity to signal for storage
        signal = Signal(  # noqa: F841
            symbol=opportunity.symbol,
            type=opportunity.signal_type,
            timestamp=opportunity.timestamp,
            confidence=opportunity.confidence,
            source=opportunity.strategy_name,
            timeframe=opportunity.timeframe,
            price=opportunity.entry_price,
            indicators=opportunity.indicators,
            metadata={**opportunity.metadata, "priority": opportunity.priority},
        )

        # Record to database if available
        if self.database:
            try:
                # This would depend on actual database implementation
                # Using a hypothetical method based on project structure
                # TODO: Implement actual database insertion
                self.logger.info(f"Recorded opportunity for {opportunity.symbol}")
            except Exception as e:
                self.logger.error(f"Failed to record opportunity: {e}")
                raise
        else:
            self.logger.warning("Database not available, opportunity not recorded")

    def get_required_market_data(
        self, symbol: str, timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data needed for opportunity evaluation.

        Args:
            symbol: Symbol to fetch data for
            timeframes: List of timeframes to fetch

        Returns:
            Dictionary of market data by timeframe

        Raises:
            DataError: If required data cannot be fetched
        """
        market_data = {}

        for timeframe in timeframes:
            try:
                # Fetch OHLCV data
                ohlcv = self.exchange_service.fetch_ohlcv(symbol, timeframe, limit=100)

                if ohlcv is None or len(ohlcv) == 0:
                    self.logger.warning(f"No data available for {symbol} on {timeframe}")
                    market_data[timeframe] = pd.DataFrame()
                    continue

                # Convert OHLCV list to DataFrame
                df = ohlcv_to_dataframe(ohlcv)
                market_data[timeframe] = df

            except ExchangeError as e:
                self.logger.warning(f"Error fetching {timeframe} data for {symbol}: {e}")
                market_data[timeframe] = pd.DataFrame()
            except Exception as e:
                self.logger.exception(f"Unexpected error fetching data for {symbol}: {e}")
                raise DataError(f"Failed to fetch market data for {symbol}: {str(e)}")

        return market_data
