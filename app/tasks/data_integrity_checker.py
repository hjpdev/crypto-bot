"""
Data integrity checking task for cryptocurrency market data.

This module provides functionality for checking data continuity,
verifying data integrity, and automating backfill operations when gaps
are detected.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pandas as pd

from app.core.scheduler import TaskScheduler
from app.services.data_collector import DataCollector


class DataIntegrityChecker:
    """
    Periodically checks for gaps in collected data, verifies data consistency,
    and triggers backfill operations when needed.

    Features:
    - Automated gap detection across multiple timeframes
    - Consistency checking between related timeframes
    - Verification of calculated indicator values
    - Automated backfill scheduling
    - Configurable check intervals and lookback periods
    """

    def __init__(
        self,
        data_collector: DataCollector,
        data_storage,
        config_service,
        check_interval_hours: int = 6,
        lookback_days: Dict[str, int] = None,
        backfill_missing: bool = True,
        max_backfill_days: int = 30,
        check_indicators: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the DataIntegrityChecker.

        Args:
            data_collector: Data collector service for backfilling
            data_storage: Data storage service for retrieving data
            config_service: Service for accessing system configuration
            check_interval_hours: How often to run integrity checks in hours
            lookback_days: Dictionary mapping timeframes to lookback days
            backfill_missing: Whether to automatically backfill missing data
            max_backfill_days: Maximum number of days to backfill
            check_indicators: Whether to check for missing indicator values
            logger: Logger instance for logging events
        """
        self._data_collector = data_collector
        self._data_storage = data_storage
        self._config_service = config_service
        self._check_interval_hours = check_interval_hours
        self._backfill_missing = backfill_missing
        self._max_backfill_days = max_backfill_days
        self._check_indicators = check_indicators
        self._logger = logger or logging.getLogger(__name__)

        # Default lookback periods if not provided
        self._lookback_days = lookback_days or {
            "1m": 1,  # 1 day for 1-minute data
            "5m": 3,  # 3 days for 5-minute data
            "15m": 5,  # 5 days for 15-minute data
            "1h": 14,  # 14 days for 1-hour data
            "4h": 30,  # 30 days for 4-hour data
            "1d": 90,  # 90 days for daily data
        }

        # Track active symbols
        self._active_symbols: Set[str] = set()

        # Integrity check statistics
        self._integrity_stats: Dict[str, Dict[str, Any]] = {}

        # Initialize
        self._logger.info(f"DataIntegrityChecker initialized with {check_interval_hours}h interval")

    def register_with_scheduler(self, scheduler: TaskScheduler) -> str:
        """
        Register the integrity check task with the scheduler.

        Args:
            scheduler: Task scheduler instance

        Returns:
            Task ID of the registered task
        """
        task_id = scheduler.add_task(
            name="data_integrity_check",
            interval_minutes=self._check_interval_hours * 60,
            task_func=self.run,
            description="Checks data integrity and triggers backfills",
            enabled=True,
        )

        self._logger.info(f"Registered data integrity check task with ID: {task_id}")
        return task_id

    def add_symbols(self, symbols: List[str]) -> None:
        """
        Add symbols to the active monitoring list.

        Args:
            symbols: List of trading symbols to monitor
        """
        for symbol in symbols:
            if symbol not in self._active_symbols:
                self._active_symbols.add(symbol)
                self._logger.info(f"Added {symbol} to integrity monitoring")

    def remove_symbols(self, symbols: List[str]) -> None:
        """
        Remove symbols from the active monitoring list.

        Args:
            symbols: List of trading symbols to remove
        """
        for symbol in symbols:
            if symbol in self._active_symbols:
                self._active_symbols.remove(symbol)
                self._logger.info(f"Removed {symbol} from integrity monitoring")

    def get_active_symbols(self) -> List[str]:
        """
        Get the list of currently monitored symbols.

        Returns:
            List of symbols being monitored
        """
        return list(self._active_symbols)

    def get_integrity_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about integrity checks.

        Returns:
            Dictionary with integrity check statistics
        """
        return self._integrity_stats

    def run(self) -> bool:
        """
        Run a complete integrity check on all active symbols.

        This checks data continuity for all timeframes, verifies indicator values,
        and optionally triggers backfill operations.

        Returns:
            True if the check completed successfully, False otherwise
        """
        try:
            self._logger.info("Starting data integrity check run")

            # If no active symbols, check system configuration
            if not self._active_symbols:
                try:
                    configured_symbols = self._config_service.get_trading_symbols()
                    if configured_symbols:
                        self.add_symbols(configured_symbols)
                    else:
                        self._logger.warning("No symbols configured for monitoring")
                        return True
                except Exception as e:
                    self._logger.error(f"Failed to get configured symbols: {str(e)}")
                    return False

            # Track overall success
            overall_success = True

            # Check data continuity for each symbol and timeframe
            for symbol in self._active_symbols:
                symbol_success = True  # Track success for this symbol
                for timeframe, lookback_days in self._lookback_days.items():
                    try:
                        self._logger.info(
                            f"Checking data continuity for {symbol} on {timeframe} "
                            f"with {lookback_days} day lookback"
                        )

                        result = self.check_ohlcv_continuity([symbol], timeframe, lookback_days)

                        # Update integrity stats
                        key = f"{symbol}:{timeframe}"
                        self._integrity_stats[key] = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "last_check": datetime.utcnow(),
                            "status": (
                                "success"
                                if result.get(symbol, {}).get("status") != "error"
                                else "error"
                            ),
                            "gaps_found": result.get(symbol, {}).get("gaps_found", 0),
                        }

                        # Check for errors in the result
                        if result.get(symbol, {}).get("status") == "error":
                            symbol_success = False

                        # If backfill is enabled and gaps were found
                        if (
                            self._backfill_missing
                            and result.get(symbol, {}).get("status") == "gaps_found"
                        ):
                            self._logger.info(f"Triggering backfill for {symbol} on {timeframe}")
                            backfill_result = self._data_collector.backfill_missing_data(
                                [symbol], timeframe
                            )

                            # Update stats with backfill result
                            self._integrity_stats[key]["backfill_status"] = backfill_result.get(
                                symbol, {}
                            ).get("status")

                            # Check for errors in backfill
                            if backfill_result.get(symbol, {}).get("status") == "error":
                                symbol_success = False

                    except Exception as e:
                        symbol_success = False
                        overall_success = False
                        self._logger.exception(
                            f"Error checking continuity for {symbol} on {timeframe}: {str(e)}"
                        )
                        # Update stats with error
                        key = f"{symbol}:{timeframe}"
                        self._integrity_stats[key] = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "last_check": datetime.utcnow(),
                            "status": "error",
                            "error": str(e),
                        }

                # Optionally verify indicator values
                if self._check_indicators:
                    try:
                        indicator_result = self.verify_indicator_values([symbol])
                        # Check for errors in any timeframe
                        for tf, tf_result in indicator_result.get(symbol, {}).items():
                            if tf_result.get("status") == "error":
                                symbol_success = False
                    except Exception as e:
                        symbol_success = False
                        overall_success = False
                        self._logger.exception(f"Error verifying indicators for {symbol}: {str(e)}")

                # Update overall success
                if not symbol_success:
                    overall_success = False

            self._logger.info("Completed data integrity check run")
            return overall_success

        except Exception as e:
            self._logger.exception(f"Error during integrity check run: {str(e)}")
            return False

    def check_ohlcv_continuity(
        self,
        symbols: List[str],
        timeframe: str,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check for continuity in OHLCV data for specified symbols.

        Args:
            symbols: List of trading symbols to check
            timeframe: Timeframe to check (e.g., "1h")
            lookback_days: Number of days to look back (None for default)

        Returns:
            Dictionary with results for each symbol
        """
        results = {}

        # Use default lookback if not specified
        if lookback_days is None:
            lookback_days = self._lookback_days.get(timeframe, 14)

        # Calculate the start time for the check
        start_time = datetime.utcnow() - timedelta(days=lookback_days)
        end_time = datetime.utcnow()

        for symbol in symbols:
            try:
                # Check data continuity using the data storage service
                gaps = self._data_storage.check_data_continuity(
                    symbol,
                    timeframe,
                    start_time,
                    end_time,
                )

                if not gaps:
                    self._logger.info(f"No gaps found for {symbol} on {timeframe}")
                    results[symbol] = {
                        "status": "no_gaps",
                        "gaps_found": 0,
                        "timeframe": timeframe,
                        "lookback_days": lookback_days,
                    }
                else:
                    gap_count = len(gaps)
                    total_gap_seconds = sum((end - start).total_seconds() for start, end in gaps)

                    self._logger.warning(
                        f"Found {gap_count} gaps in {symbol} on {timeframe}, "
                        f"total missing time: {total_gap_seconds/3600:.2f} hours"
                    )

                    results[symbol] = {
                        "status": "gaps_found",
                        "gaps_found": gap_count,
                        "timeframe": timeframe,
                        "lookback_days": lookback_days,
                        "total_gap_seconds": total_gap_seconds,
                        "gap_periods": [
                            {"start": start.isoformat(), "end": end.isoformat()}
                            for start, end in gaps
                        ],
                    }

            except Exception as e:
                self._logger.exception(
                    f"Error checking continuity for {symbol} on {timeframe}: {str(e)}"
                )
                results[symbol] = {
                    "status": "error",
                    "timeframe": timeframe,
                    "lookback_days": lookback_days,
                    "error": str(e),
                    "gaps_found": 0,
                }

        return results

    def verify_indicator_values(
        self,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        required_indicators: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify that indicator values have been calculated for OHLCV data.

        Args:
            symbols: List of trading symbols to check
            timeframes: List of timeframes to check (None for all monitored)
            required_indicators: List of required indicator names (None for any)

        Returns:
            Dictionary with verification results for each symbol
        """
        results = {}

        # Use default timeframes if not specified
        if timeframes is None:
            timeframes = list(self._lookback_days.keys())

        for symbol in symbols:
            symbol_results = {}

            for timeframe in timeframes:
                try:
                    # Get the last N records with indicators included
                    lookback_days = self._lookback_days.get(timeframe, 14)
                    start_time = datetime.utcnow() - timedelta(days=lookback_days)

                    # Get the data as a DataFrame for easier analysis
                    df = self._data_storage.get_ohlcv(
                        symbol,
                        timeframe,
                        start_time=start_time,
                        include_indicators=True,
                        as_dataframe=True,
                    )

                    if df.empty:
                        self._logger.warning(
                            f"No data found for {symbol} on {timeframe} for indicator verification"
                        )
                        symbol_results[timeframe] = {
                            "status": "no_data",
                            "message": "No data available for indicator verification",
                        }
                        continue

                    # Check for indicators
                    # Filter out OHLCV columns to get just indicator columns
                    ohlcv_columns = ["open", "high", "low", "close", "volume"]
                    indicator_columns = [col for col in df.columns if col not in ohlcv_columns]

                    if not indicator_columns:
                        self._logger.warning(f"No indicators found for {symbol} on {timeframe}")
                        symbol_results[timeframe] = {
                            "status": "no_indicators",
                            "message": "No indicators found in the data",
                        }
                        continue

                    # If specific indicators are required, check for them
                    missing_indicators = []
                    if required_indicators:
                        missing_indicators = [
                            ind for ind in required_indicators if ind not in indicator_columns
                        ]

                        if missing_indicators:
                            self._logger.warning(
                                f"Missing required indicators for {symbol} on {timeframe}: "
                                f"{', '.join(missing_indicators)}"
                            )
                            symbol_results[timeframe] = {
                                "status": "missing_indicators",
                                "missing": missing_indicators,
                                "available": indicator_columns,
                            }
                            continue

                    # Check for missing values (NaN) in indicators
                    nan_counts = df[indicator_columns].isna().sum().to_dict()
                    total_rows = len(df)

                    # Calculate percentage of missing values
                    nan_percentages = {
                        ind: count / total_rows * 100
                        for ind, count in nan_counts.items()
                        if count > 0
                    }

                    if nan_percentages:
                        self._logger.warning(
                            f"Found missing indicator values for {symbol} on {timeframe}: "
                            f"{nan_percentages}"
                        )
                        symbol_results[timeframe] = {
                            "status": "incomplete_indicators",
                            "nan_percentages": nan_percentages,
                            "total_rows": total_rows,
                            "available_indicators": indicator_columns,
                        }
                    else:
                        self._logger.info(f"All indicators complete for {symbol} on {timeframe}")
                        symbol_results[timeframe] = {
                            "status": "complete",
                            "available_indicators": indicator_columns,
                            "total_rows": total_rows,
                        }

                except Exception as e:
                    self._logger.exception(
                        f"Error verifying indicators for {symbol} on {timeframe}: {str(e)}"
                    )
                    symbol_results[timeframe] = {
                        "status": "error",
                        "error": str(e),
                    }

            results[symbol] = symbol_results

        return results

    def check_cross_timeframe_consistency(
        self,
        symbols: List[str],
        base_timeframe: str = "1h",
        derived_timeframes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check consistency between different timeframes of the same symbol.

        This verifies that data in larger timeframes (e.g., 1h) is consistent
        with data in smaller timeframes (e.g., 1m) when aggregated.

        Args:
            symbols: List of trading symbols to check
            base_timeframe: Base timeframe to check against
            derived_timeframes: Larger timeframes to check (None for default)

        Returns:
            Dictionary with consistency check results
        """
        results = {}

        # Default derived timeframes if not specified
        if derived_timeframes is None:
            # Determine derived timeframes based on base_timeframe
            if base_timeframe == "1m":
                derived_timeframes = ["5m", "15m", "1h"]
            elif base_timeframe == "1h":
                derived_timeframes = ["4h", "1d"]
            else:
                derived_timeframes = []

        # Set thresholds for acceptable differences (percentage)
        # These are configurable for different types of data
        threshold = 5.0  # 5% difference for price data
        volume_threshold = 150.0  # 150% for volume (more variable)

        for symbol in symbols:
            symbol_results = {}

            try:
                # Get base timeframe data
                lookback_days = self._lookback_days.get(base_timeframe, 7)
                start_time = datetime.utcnow() - timedelta(days=lookback_days)

                base_df = self._data_storage.get_ohlcv(
                    symbol,
                    base_timeframe,
                    start_time=start_time,
                    as_dataframe=True,
                )

                if base_df.empty:
                    self._logger.warning(
                        f"No data found for {symbol} on {base_timeframe} for consistency check"
                    )
                    results[symbol] = {
                        "status": "no_data",
                        "message": f"No data available for {base_timeframe}",
                    }
                    continue

                # Check each derived timeframe
                for derived_tf in derived_timeframes:
                    derived_df = self._data_storage.get_ohlcv(
                        symbol,
                        derived_tf,
                        start_time=start_time,
                        as_dataframe=True,
                    )

                    if derived_df.empty:
                        self._logger.warning(
                            f"No data found for {symbol} on {derived_tf} for consistency check"
                        )
                        symbol_results[derived_tf] = {
                            "status": "no_data",
                            "message": f"No data available for {derived_tf}",
                        }
                        continue

                    # Resample base data to derived timeframe for comparison
                    # This resampling logic will depend on the specific timeframe formats
                    # and pandas resample rules
                    resample_rule = self._timeframe_to_resample_rule(derived_tf)

                    if not resample_rule:
                        symbol_results[derived_tf] = {
                            "status": "error",
                            "error": f"Cannot determine resample rule for {derived_tf}",
                        }
                        continue

                    # Resample the base timeframe to match the derived timeframe
                    resampled = base_df.resample(resample_rule).agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )

                    # Check for matching timestamps
                    common_timestamps = derived_df.index.intersection(resampled.index)
                    if len(common_timestamps) == 0:
                        self._logger.warning(
                            f"No common timestamps found between {base_timeframe} and {derived_tf} "
                            f"for {symbol}"
                        )
                        symbol_results[derived_tf] = {
                            "status": "no_common_timestamps",
                            "message": "No common timestamps found between timeframes",
                        }
                        continue

                    # Compare values
                    derived_slice = derived_df.loc[common_timestamps]
                    resampled_slice = resampled.loc[common_timestamps]

                    # Calculate differences as percentage
                    # Handle zeros in resampled data to avoid division by zero
                    diff_pct = pd.DataFrame()
                    for col in ["open", "high", "low", "close", "volume"]:
                        # Replace zeros with NaN to avoid division by zero
                        denominator = resampled_slice[col].replace(0, float("nan"))
                        diff_pct[col] = (
                            100 * abs(derived_slice[col] - resampled_slice[col]) / denominator
                        )

                    # Check if differences exceed thresholds
                    max_diff_pct = {
                        "open": diff_pct["open"].max() if not diff_pct["open"].empty else 0,
                        "high": diff_pct["high"].max() if not diff_pct["high"].empty else 0,
                        "low": diff_pct["low"].max() if not diff_pct["low"].empty else 0,
                        "close": diff_pct["close"].max() if not diff_pct["close"].empty else 0,
                        "volume": diff_pct["volume"].max() if not diff_pct["volume"].empty else 0,
                    }

                    avg_diff_pct = {
                        "open": diff_pct["open"].mean() if not diff_pct["open"].empty else 0,
                        "high": diff_pct["high"].mean() if not diff_pct["high"].empty else 0,
                        "low": diff_pct["low"].mean() if not diff_pct["low"].empty else 0,
                        "close": diff_pct["close"].mean() if not diff_pct["close"].empty else 0,
                        "volume": diff_pct["volume"].mean() if not diff_pct["volume"].empty else 0,
                    }

                    # Check if any exceed thresholds
                    issues = []
                    if max_diff_pct["open"] > threshold:
                        issues.append(f"open max diff: {max_diff_pct['open']:.2f}%")
                    if max_diff_pct["high"] > threshold:
                        issues.append(f"high max diff: {max_diff_pct['high']:.2f}%")
                    if max_diff_pct["low"] > threshold:
                        issues.append(f"low max diff: {max_diff_pct['low']:.2f}%")
                    if max_diff_pct["close"] > threshold:
                        issues.append(f"close max diff: {max_diff_pct['close']:.2f}%")
                    if max_diff_pct["volume"] > volume_threshold:
                        issues.append(f"volume max diff: {max_diff_pct['volume']:.2f}%")

                    if issues:
                        self._logger.warning(
                            f"Consistency issues between {base_timeframe} and {derived_tf} "
                            f"for {symbol}: {', '.join(issues)}"
                        )
                        symbol_results[derived_tf] = {
                            "status": "inconsistent",
                            "issues": issues,
                            "max_diff_pct": max_diff_pct,
                            "avg_diff_pct": avg_diff_pct,
                            "common_timestamps": len(common_timestamps),
                        }
                    else:
                        self._logger.info(
                            f"Consistency check passed between {base_timeframe} and {derived_tf} "
                            f"for {symbol}"
                        )
                        symbol_results[derived_tf] = {
                            "status": "consistent",
                            "max_diff_pct": max_diff_pct,
                            "avg_diff_pct": avg_diff_pct,
                            "common_timestamps": len(common_timestamps),
                        }

            except Exception as e:
                self._logger.exception(
                    f"Error during cross-timeframe consistency check for {symbol}: {str(e)}"
                )
                symbol_results["error"] = {
                    "status": "error",
                    "error": str(e),
                }

            results[symbol] = symbol_results

        return results

    def _timeframe_to_resample_rule(self, timeframe: str) -> Optional[str]:
        """
        Convert a timeframe string to a pandas resample rule.

        Args:
            timeframe: Timeframe string (e.g., "1h", "15m", "1d")

        Returns:
            Pandas resample rule or None if invalid
        """
        if not timeframe:
            return None

        # Parse the number and unit from the timeframe
        unit = timeframe[-1].lower()
        try:
            number = int(timeframe[:-1])
        except ValueError:
            return None

        # Convert to pandas resample rule
        if unit == "m":
            return f"{number}min"
        elif unit == "h":
            return f"{number}H"
        elif unit == "d":
            return f"{number}D"
        elif unit == "w":
            return f"{number}W"
        else:
            return None

    def export_integrity_stats(self) -> pd.DataFrame:
        """
        Export integrity check statistics as a DataFrame.

        Returns:
            DataFrame containing integrity statistics
        """
        stats_list = list(self._integrity_stats.values())
        if not stats_list:
            return pd.DataFrame()

        return pd.DataFrame(stats_list)
