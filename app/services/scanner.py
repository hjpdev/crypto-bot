"""
Opportunity scanner service for identifying trading opportunities.

This module provides functionality for scanning markets and identifying
potential trading opportunities based on configured strategies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, validator

from app.core.database import Database
from app.core.exceptions import DataError, ExchangeError, DatabaseError
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
    
    @validator('confidence')
    def check_confidence_range(cls, v):
        """Validate that confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


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
        if strategy is None:
            self.logger.warning("Attempted to add None strategy, ignoring")
            return
            
        # Check if strategy already exists
        strategy_name = strategy.__class__.__name__
        if any(s.__class__.__name__ == strategy_name for s in self.strategies):
            self.logger.warning(f"Strategy {strategy_name} already added, ignoring")
            return
            
        self.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy_name}")

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
        if not symbols:
            self.logger.warning("No symbols provided for scanning")
            return []
            
        timeframes = timeframes or self.default_timeframes
        all_opportunities = []

        self.logger.info(f"Scanning {len(symbols)} markets across {len(timeframes)} timeframes")

        for symbol in symbols:
            try:
                # Get market data for the symbol
                market_data = self._get_market_data(symbol, timeframes)

                # Evaluate opportunities for this symbol
                symbol_opportunities = self._evaluate_symbol_opportunities(symbol, market_data)
                all_opportunities.extend(symbol_opportunities)

            except (ExchangeError, DataError) as e:
                self.logger.warning(f"Error scanning market {symbol}: {e}")
                continue
            except Exception as e:
                self.logger.exception(f"Unexpected error scanning market {symbol}: {e}")
                continue

        # Process the found opportunities
        processed_opportunities = self._process_opportunities(all_opportunities)
        
        self.logger.info(
            f"Scan completed. Found {len(all_opportunities)} opportunities, "
            f"{len(processed_opportunities)} after processing."
        )

        return processed_opportunities
        
    def _get_market_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for a symbol across multiple timeframes.
        
        Args:
            symbol: The trading symbol to fetch data for
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary of market data by timeframe
            
        Raises:
            DataError: If market data cannot be retrieved
        """
        return self.get_required_market_data(symbol, timeframes)
        
    def _evaluate_symbol_opportunities(
        self, symbol: str, market_data: Dict[str, pd.DataFrame]
    ) -> List[Opportunity]:
        """
        Evaluate a single symbol for trading opportunities across all timeframes.
        
        Args:
            symbol: Symbol to evaluate
            market_data: Dictionary of market data by timeframe
            
        Returns:
            List of identified opportunities
        """
        return self.evaluate_opportunity(symbol, market_data)
        
    def _process_opportunities(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """
        Process the found opportunities - filter, prioritize and record.
        
        Args:
            opportunities: List of opportunities to process
            
        Returns:
            Processed list of opportunities
        """
        # Filter and prioritize opportunities
        filtered_opportunities = self.filter_opportunities(opportunities)
        prioritized_opportunities = self.prioritize_opportunities(filtered_opportunities)

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

                    # Skip if not enough data for analysis
                    min_required_rows = self._get_min_required_rows(strategy)
                    if len(data) < min_required_rows:
                        self.logger.debug(
                            f"Not enough data for {symbol} on {timeframe}, "
                            f"got {len(data)} rows, need at least {min_required_rows}"
                        )
                        continue

                    # Analyze market conditions
                    regime = self.market_analyzer.detect_market_regime(data)

                    # Apply strategy to get signals
                    opportunities_from_timeframe = self._apply_strategy(
                        strategy, symbol, timeframe, data, regime
                    )
                    opportunities.extend(opportunities_from_timeframe)

            except Exception as e:
                self.logger.exception(
                    f"Error applying strategy {strategy.__class__.__name__} to {symbol}: {e}"
                )
                continue

        return opportunities
        
    def _get_min_required_rows(self, strategy: Any) -> int:
        """
        Get the minimum required rows for a strategy based on its indicators.
        
        Args:
            strategy: The strategy to check
            
        Returns:
            Minimum number of rows required
        """
        # Default to a reasonable value if we can't determine from the strategy
        return 100
        
    def _apply_strategy(
        self, 
        strategy: Any, 
        symbol: str, 
        timeframe: str, 
        data: pd.DataFrame, 
        regime: str
    ) -> List[Opportunity]:
        """
        Apply a strategy to a single timeframe of data.
        
        Args:
            strategy: Strategy to apply
            symbol: Symbol being analyzed
            timeframe: Timeframe being analyzed
            data: OHLCV data for the timeframe
            regime: Detected market regime
            
        Returns:
            List of opportunities found
        """
        opportunities = []
        strategy_name = strategy.__class__.__name__
        
        try:
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
                opportunity = self._create_opportunity_from_signal(
                    symbol, timeframe, data, entry_info, strategy_name, regime
                )
                opportunities.append(opportunity)
                self.logger.debug(
                    f"Found opportunity for {symbol} on {timeframe}: {entry_info.get('reason', '')}"
                )
        except Exception as e:
            self.logger.exception(
                f"Error evaluating {strategy_name} on {symbol}/{timeframe}: {e}"
            )
            
        return opportunities
        
    def _create_opportunity_from_signal(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        entry_info: Dict[str, Any],
        strategy_name: str,
        regime: str
    ) -> Opportunity:
        """
        Create an Opportunity object from a signal.
        
        Args:
            symbol: Symbol being analyzed
            timeframe: Timeframe being analyzed
            data: OHLCV data for the timeframe
            entry_info: Entry information from the strategy
            strategy_name: Name of the strategy
            regime: Detected market regime
            
        Returns:
            Created Opportunity object
        """
        signal_type = (
            SignalType.BUY
            if entry_info.get("direction", "long") == "long"
            else SignalType.SELL
        )
        
        return Opportunity(
            symbol=symbol,
            signal_type=signal_type,
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
        min_confidence = 0.3  # Could be moved to a configuration parameter

        for opportunity in opportunities:
            # Filter out low confidence opportunities
            if opportunity.confidence < min_confidence:
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
            priority_score = self._calculate_priority_score(opportunity)
            opportunity.priority = priority_score

        # Sort by priority (descending)
        prioritized = sorted(opportunities, key=lambda x: x.priority, reverse=True)

        return prioritized
        
    def _calculate_priority_score(self, opportunity: Opportunity) -> int:
        """
        Calculate a priority score for an opportunity.
        
        Args:
            opportunity: The opportunity to score
            
        Returns:
            Priority score (higher is better)
        """
        priority_score = 0

        # Base score from confidence
        priority_score += int(opportunity.confidence * 100)

        # Higher priority for daily timeframe signals
        timeframe_scores = {
            "1d": 50,
            "4h": 30,
            "1h": 10,
            "15m": 5
        }
        
        priority_score += timeframe_scores.get(opportunity.timeframe, 0)

        # Add other priority factors here:
        # - Historical strategy performance
        # - Current market conditions
        # - Asset volatility
        # - Trading volume

        return priority_score

    def record_opportunity(self, opportunity: Opportunity) -> None:
        """
        Save an opportunity to the database.

        Args:
            opportunity: Opportunity to record

        Raises:
            DatabaseError: If opportunity cannot be saved
        """
        # Convert opportunity to signal for storage
        signal = self._convert_opportunity_to_signal(opportunity)

        # Record to database if available
        if self.database:
            try:
                self._store_signal_in_database(signal)
                self.logger.info(f"Recorded opportunity for {opportunity.symbol}")
            except Exception as e:
                self.logger.error(f"Failed to record opportunity: {e}")
                raise DatabaseError(f"Failed to store opportunity in database: {str(e)}")
        else:
            self.logger.warning("Database not available, opportunity not recorded")
            
    def _convert_opportunity_to_signal(self, opportunity: Opportunity) -> Signal:
        """
        Convert an Opportunity object to a Signal object for storage.
        
        Args:
            opportunity: The opportunity to convert
            
        Returns:
            Signal object
        """
        return Signal(
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
        
    def _store_signal_in_database(self, signal: Signal) -> None:
        """
        Store a signal in the database.
        
        Args:
            signal: The signal to store
            
        Raises:
            DatabaseError: If the signal cannot be stored
        """
        # TODO: Implement actual database insertion
        # This is a placeholder for the actual implementation
        pass

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
                ohlcv = self._fetch_ohlcv_data(symbol, timeframe)
                
                if not ohlcv:
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
        
    def _fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[List[float]]:
        """
        Fetch OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe to fetch
            limit: Number of candles to fetch
            
        Returns:
            OHLCV data as a list of lists or None if no data is available
            
        Raises:
            ExchangeError: If there's an error fetching data from the exchange
        """
        ohlcv = self.exchange_service.fetch_ohlcv(symbol, timeframe, limit=limit)

        if ohlcv is None or len(ohlcv) == 0:
            self.logger.warning(f"No data available for {symbol} on {timeframe}")
            return None
            
        return ohlcv
