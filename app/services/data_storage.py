"""
Data storage service for efficient cryptocurrency market data storage.

This module provides an abstraction layer for database operations optimized
for time series data storage, retrieval, and management.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.dialects.postgresql import insert
from contextlib import contextmanager

from app.core.exceptions import DatabaseError, DataError
from app.models.ohlcv import OHLCV
from app.models.cryptocurrency import Cryptocurrency
from app.models.market_snapshot import MarketSnapshot


class DataStorage:
    """
    Provides an abstraction layer for database operations, optimized for time
    series data storage, retrieval, and management.

    Features:
    - Efficient bulk insert operations for time series data
    - Optimized data storage for time series data
    - Gap detection and continuity checks
    - Transaction management
    - Clean abstraction layer between application and database
    """

    def __init__(
        self,
        session_provider,
        logger: Optional[logging.Logger] = None,
        batch_size: int = 1000,
        optimize_writes: bool = True,
    ):
        """
        Initialize the DataStorage service with necessary configuration.

        Args:
            session_provider: Callable that provides a database session
            logger: Logger instance for logging events
            batch_size: Default batch size for bulk operations
            optimize_writes: Whether to optimize write operations for performance
        """
        self._session_provider = session_provider
        self._logger = logger or logging.getLogger(__name__)
        self._batch_size = batch_size
        self._optimize_writes = optimize_writes

        self._logger.info("DataStorage service initialized")

    def store_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        ohlcv_data: List[List[Union[int, float]]],
        upsert: bool = True,
    ) -> int:
        """
        Store OHLCV data for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Timeframe of the data (e.g., "1h")
            ohlcv_data: List of OHLCV candles from exchange
            upsert: Whether to update existing records if they exist

        Returns:
            Number of records stored

        Raises:
            DatabaseError: If there is an issue with the database operation
            ValidationError: If data validation fails
        """
        if not ohlcv_data:
            self._logger.warning(f"No OHLCV data provided for {symbol} on {timeframe}")
            return 0

        try:
            session = self.get_session()

            # Get or create cryptocurrency record
            crypto = self._get_or_create_cryptocurrency(session, symbol)

            # Parse exchange from symbol
            exchange, _ = self._parse_symbol(symbol)

            # Prepare records for bulk insert
            records = []
            for candle in ohlcv_data:
                # Standard OHLCV format: [timestamp, open, high, low, close, volume]
                if len(candle) < 6:
                    self._logger.warning(f"Skipping incomplete OHLCV candle: {candle}")
                    continue

                timestamp_ms, open_price, high_price, low_price, close_price, volume = candle[:6]

                # Convert timestamp from milliseconds to datetime
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)

                # Create OHLCV record
                record = {
                    "cryptocurrency_id": crypto.id,
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
                records.append(record)

            if not records:
                self._logger.warning(f"No valid OHLCV records to store for {symbol} on {timeframe}")
                return 0

            # Use bulk insert or upsert
            if upsert:
                return self._upsert_ohlcv_records(session, records)
            else:
                return self._insert_ohlcv_records(session, records)

        except SQLAlchemyError as e:
            self._logger.exception(f"Database error storing OHLCV data for {symbol}: {str(e)}")
            raise DatabaseError(f"Failed to store OHLCV data: {str(e)}") from e
        except Exception as e:
            self._logger.exception(f"Unexpected error storing OHLCV data for {symbol}: {str(e)}")
            raise DataError(f"Failed to store OHLCV data: {str(e)}") from e

    def store_indicator_values(
        self,
        symbol: str,
        timeframe: str,
        indicators: Dict[datetime, Dict[str, Any]],
        merge_existing: bool = True,
    ) -> int:
        """
        Store calculated indicator values for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Timeframe of the data (e.g., "1h")
            indicators: Dictionary mapping timestamps to indicator values
            merge_existing: Whether to merge with existing indicator values

        Returns:
            Number of records updated

        Raises:
            DatabaseError: If there is an issue with the database operation
        """
        if not indicators:
            self._logger.warning(f"No indicator data provided for {symbol} on {timeframe}")
            return 0

        try:
            session = self.get_session()

            # Get or create cryptocurrency record
            crypto = self._get_or_create_cryptocurrency(session, symbol)

            count = 0
            for timestamp, indicator_values in indicators.items():
                # Find the OHLCV record for this timestamp
                ohlcv = (
                    session.query(OHLCV)
                    .filter(
                        OHLCV.cryptocurrency_id == crypto.id,
                        OHLCV.timeframe == timeframe,
                        OHLCV.timestamp == timestamp,
                    )
                    .first()
                )

                if not ohlcv:
                    self._logger.warning(
                        f"No OHLCV record found for {symbol} on {timeframe} at {timestamp}, "
                        "cannot store indicators"
                    )
                    continue

                # Update indicators
                if merge_existing and ohlcv.indicators:
                    # Merge with existing indicators
                    current_indicators = ohlcv.indicators or {}
                    current_indicators.update(indicator_values)
                    ohlcv.indicators = current_indicators
                else:
                    # Replace existing indicators
                    ohlcv.indicators = indicator_values

                count += 1

                # Commit in batches
                if self._optimize_writes and count % self._batch_size == 0:
                    session.commit()

            # Final commit
            session.commit()
            self._logger.info(f"Updated indicators for {count} records for {symbol} on {timeframe}")
            return count

        except SQLAlchemyError as e:
            self._logger.exception(f"Database error storing indicators for {symbol}: {str(e)}")
            raise DatabaseError(f"Failed to store indicator values: {str(e)}") from e
        except Exception as e:
            self._logger.exception(f"Unexpected error storing indicators for {symbol}: {str(e)}")
            raise DataError(f"Failed to store indicator values: {str(e)}") from e

    def store_market_snapshot(
        self,
        symbol: str,
        snapshot: Dict[str, Any],
    ) -> int:
        """
        Store a comprehensive market snapshot for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            snapshot: Market snapshot data

        Returns:
            ID of the stored snapshot

        Raises:
            DatabaseError: If there is an issue with the database operation
            ValidationError: If data validation fails
        """
        if not snapshot:
            self._logger.warning(f"Empty snapshot provided for {symbol}")
            return 0

        try:
            session = self.get_session()

            # Get or create cryptocurrency record
            crypto = self._get_or_create_cryptocurrency(session, symbol)

            # Parse exchange from symbol
            exchange, _ = self._parse_symbol(symbol)

            # Create snapshot record
            timestamp = snapshot.get("timestamp", datetime.utcnow())

            # Extract data from snapshot
            data = {}
            if "ticker" in snapshot and snapshot["ticker"]:
                data["ticker"] = snapshot["ticker"]

            if "order_book" in snapshot and snapshot["order_book"]:
                data["order_book"] = snapshot["order_book"]

            if "trades" in snapshot and snapshot["trades"]:
                data["trades"] = snapshot["trades"]

            snapshot_record = MarketSnapshot(
                cryptocurrency_id=crypto.id,
                exchange=exchange,
                symbol=symbol,
                timestamp=timestamp,
                data=data,
            )

            session.add(snapshot_record)
            session.commit()

            self._logger.info(f"Stored market snapshot for {symbol} at {timestamp}")
            return snapshot_record.id

        except SQLAlchemyError as e:
            self._logger.exception(f"Database error storing market snapshot for {symbol}: {str(e)}")
            raise DatabaseError(f"Failed to store market snapshot: {str(e)}") from e
        except Exception as e:
            self._logger.exception(
                f"Unexpected error storing market snapshot for {symbol}: {str(e)}"
            )
            raise DataError(f"Failed to store market snapshot: {str(e)}") from e

    def store_order_book(
        self,
        symbol: str,
        order_book: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Store an order book snapshot for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            order_book: Order book data
            timestamp: Timestamp of the snapshot (defaults to current time)

        Returns:
            ID of the stored order book

        Raises:
            DatabaseError: If there is an issue with the database operation
        """
        timestamp = timestamp or datetime.utcnow()

        try:
            # Create market snapshot with order book data
            snapshot_data = {
                "timestamp": timestamp,
                "order_book": order_book,
            }

            return self.store_market_snapshot(symbol, snapshot_data)

        except Exception as e:
            self._logger.exception(f"Error storing order book for {symbol}: {str(e)}")
            raise DataError(f"Failed to store order book: {str(e)}") from e

    def bulk_insert(
        self,
        model_class,
        records: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        return_ids: bool = False,
    ) -> Union[int, List[int]]:
        """
        Perform an efficient bulk insert of records.

        Args:
            model_class: SQLAlchemy model class
            records: List of record dictionaries to insert
            batch_size: Size of batches to insert (defaults to class batch_size)
            return_ids: Whether to return the IDs of inserted records

        Returns:
            Number of inserted records or list of inserted record IDs

        Raises:
            DatabaseError: If there is an issue with the database operation
        """
        if not records:
            return 0 if not return_ids else []

        batch_size = batch_size or self._batch_size
        inserted_ids = [] if return_ids else None

        try:
            session = self.get_session()

            total_count = 0

            # Process in batches
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]

                if return_ids:
                    # Need to use a different approach if we need the IDs
                    for record in batch:
                        instance = model_class(**record)
                        session.add(instance)
                        session.flush()  # Flush to get the ID
                        inserted_ids.append(instance.id)
                else:
                    # Use the fastest bulk insert method
                    session.bulk_insert_mappings(model_class, batch)

                total_count += len(batch)

                # Commit each batch if optimizing writes
                if self._optimize_writes:
                    session.commit()

            # Final commit if not already committed
            if not self._optimize_writes:
                session.commit()

            self._logger.info(f"Bulk inserted {total_count} records of type {model_class.__name__}")
            return inserted_ids if return_ids else total_count

        except SQLAlchemyError as e:
            self._logger.exception(
                f"Database error during bulk insert of {model_class.__name__}: {str(e)}"
            )
            raise DatabaseError(f"Failed to perform bulk insert: {str(e)}") from e
        except Exception as e:
            self._logger.exception(
                f"Unexpected error during bulk insert of {model_class.__name__}: {str(e)}"
            )
            raise DataError(f"Failed to perform bulk insert: {str(e)}") from e

    def check_data_continuity(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_gap_multiplier: float = 1.5,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Check for gaps in time series data for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Timeframe of the data (e.g., "1h")
            start_time: Start time for the check (None for earliest available)
            end_time: End time for the check (None for latest available)
            max_gap_multiplier: Factor to determine what constitutes a gap

        Returns:
            List of (start, end) datetime tuples representing gaps

        Raises:
            DatabaseError: If there is an issue with the database operation
        """
        try:
            session = self.get_session()

            # Get cryptocurrency
            crypto = self._get_cryptocurrency(session, symbol)
            if not crypto:
                self._logger.warning(f"No cryptocurrency record found for {symbol}")
                return []

            # Determine the interval in seconds for the timeframe
            interval_seconds = self._timeframe_to_seconds(timeframe)
            max_gap_seconds = interval_seconds * max_gap_multiplier

            # Query OHLCV data for the time range
            query = (
                session.query(OHLCV)
                .filter(OHLCV.cryptocurrency_id == crypto.id, OHLCV.timeframe == timeframe)
                .order_by(OHLCV.timestamp.asc())
            )

            if start_time:
                query = query.filter(OHLCV.timestamp >= start_time)
            if end_time:
                query = query.filter(OHLCV.timestamp <= end_time)

            # Execute query and process results
            ohlcv_data = query.all()

            if not ohlcv_data:
                self._logger.warning(f"No OHLCV data found for {symbol} on {timeframe}")
                # The entire range is a gap
                if start_time and end_time:
                    return [(start_time, end_time)]
                return []

            # Initialize with the actual range boundaries
            range_start = start_time or ohlcv_data[0].timestamp
            range_end = end_time or ohlcv_data[-1].timestamp

            # Find gaps
            gaps = []
            prev_timestamp = None

            for record in ohlcv_data:
                current_timestamp = record.timestamp

                # Check for gap at the beginning
                if (
                    prev_timestamp is None
                    and range_start
                    and current_timestamp - range_start > timedelta(seconds=max_gap_seconds)
                ):
                    gaps.append((range_start, current_timestamp))

                # Check for gap between records
                elif (
                    prev_timestamp
                    and (current_timestamp - prev_timestamp).total_seconds() > max_gap_seconds
                ):
                    gaps.append((prev_timestamp, current_timestamp))

                prev_timestamp = current_timestamp

            # Check for gap at the end
            if (
                prev_timestamp
                and range_end
                and range_end - prev_timestamp > timedelta(seconds=max_gap_seconds)
            ):
                gaps.append((prev_timestamp, range_end))

            self._logger.info(f"Found {len(gaps)} gaps in data for {symbol} on {timeframe}")
            return gaps

        except SQLAlchemyError as e:
            self._logger.exception(
                f"Database error checking data continuity for {symbol}: {str(e)}"
            )
            raise DatabaseError(f"Failed to check data continuity: {str(e)}") from e
        except Exception as e:
            self._logger.exception(
                f"Unexpected error checking data continuity for {symbol}: {str(e)}"
            )
            raise DataError(f"Failed to check data continuity: {str(e)}") from e

    @contextmanager
    def _session_context(self) -> Session:
        """
        Get a database session as a context manager.

        Returns:
            SQLAlchemy session object
        """
        session = self._session_provider()
        try:
            yield session
        finally:
            # No need to close the session as it's managed by the provider
            pass

    def get_session(self) -> Session:
        """
        Get a database session.

        Returns:
            SQLAlchemy session object
        """
        return self._session_provider()

    def _insert_ohlcv_records(self, session: Session, records: List[Dict[str, Any]]) -> int:
        """
        Insert OHLCV records using bulk insert.

        Args:
            session: Database session
            records: OHLCV records to insert

        Returns:
            Number of records inserted
        """
        try:
            # Use bulk insert for better performance
            session.bulk_insert_mappings(OHLCV, records)
            session.commit()
            return len(records)
        except IntegrityError as e:
            # Handle duplicate key errors
            session.rollback()
            self._logger.warning(
                f"Integrity error during OHLCV insert, falling back to individual inserts: {e}"
            )

            # Fall back to individual inserts, skipping duplicates
            count = 0
            for record in records:
                try:
                    ohlcv = OHLCV(**record)
                    session.add(ohlcv)
                    session.flush()
                    count += 1
                except IntegrityError:
                    session.rollback()
                    # Skip duplicates
                    continue

            session.commit()
            return count

    def _upsert_ohlcv_records(self, session: Session, records: List[Dict[str, Any]]) -> int:
        """
        Upsert OHLCV records, updating existing ones and inserting new ones.

        Args:
            session: Database session
            records: OHLCV records to upsert

        Returns:
            Number of records affected
        """
        try:
            count = 0

            # PostgreSQL-specific upsert implementation
            try:
                # Prepare unique key fields
                unique_key_fields = ["cryptocurrency_id", "timeframe", "timestamp"]

                # Prepare update fields
                update_fields = ["open", "high", "low", "close", "volume"]

                # Batch processing
                for i in range(0, len(records), self._batch_size):
                    batch = records[i : i + self._batch_size]

                    # Use PostgreSQL upsert (INSERT ... ON CONFLICT DO UPDATE)
                    stmt = insert(OHLCV).values(batch)

                    # Define the update part
                    update_dict = {field: getattr(stmt.excluded, field) for field in update_fields}

                    # Define the conflict part
                    stmt = stmt.on_conflict_do_update(
                        index_elements=unique_key_fields, set_=update_dict
                    )

                    result = session.execute(stmt)
                    count += result.rowcount

                    # Commit each batch if optimizing writes
                    if self._optimize_writes:
                        session.commit()

                # Final commit if not already committed
                if not self._optimize_writes:
                    session.commit()

                return count

            except Exception as pg_error:
                # Fall back to a more generic approach for non-PostgreSQL databases
                self._logger.warning(f"PostgreSQL upsert not available, falling back: {pg_error}")
                session.rollback()

                # Process each record individually
                for record in records:
                    crypto_id = record["cryptocurrency_id"]
                    timeframe = record["timeframe"]
                    timestamp = record["timestamp"]

                    # Try to find existing record
                    existing = (
                        session.query(OHLCV)
                        .filter(
                            OHLCV.cryptocurrency_id == crypto_id,
                            OHLCV.timeframe == timeframe,
                            OHLCV.timestamp == timestamp,
                        )
                        .first()
                    )

                    if existing:
                        # Update existing record
                        existing.open = record["open"]
                        existing.high = record["high"]
                        existing.low = record["low"]
                        existing.close = record["close"]
                        existing.volume = record["volume"]
                        count += 1
                    else:
                        # Insert new record
                        ohlcv = OHLCV(**record)
                        session.add(ohlcv)
                        count += 1

                    # Commit periodically
                    if self._optimize_writes and count % self._batch_size == 0:
                        session.commit()

                # Final commit
                session.commit()
                return count

        except SQLAlchemyError as e:
            session.rollback()
            self._logger.exception(f"Database error during OHLCV upsert: {str(e)}")
            raise DatabaseError(f"Failed to upsert OHLCV records: {str(e)}") from e

    def _get_cryptocurrency(self, session: Session, symbol: str) -> Optional[Cryptocurrency]:
        """
        Get a cryptocurrency record for a symbol.

        Args:
            session: Database session
            symbol: Trading symbol (e.g., "BTC/USD")

        Returns:
            Cryptocurrency record or None if not found
        """
        # Parse exchange and base symbol
        _, parsed_symbol = self._parse_symbol(symbol)

        return session.query(Cryptocurrency).filter(Cryptocurrency.symbol == parsed_symbol).first()

    def _get_or_create_cryptocurrency(self, session: Session, symbol: str) -> Cryptocurrency:
        """
        Get or create a cryptocurrency record for a symbol.

        Args:
            session: Database session
            symbol: Trading symbol (e.g., "BTC/USD")

        Returns:
            Cryptocurrency record
        """
        # Parse exchange and base symbol
        _, parsed_symbol = self._parse_symbol(symbol)

        # Try to find existing record
        crypto = (
            session.query(Cryptocurrency).filter(Cryptocurrency.symbol == parsed_symbol).first()
        )

        if not crypto:
            # Create new record
            symbol_parts = parsed_symbol.split("/")
            name = symbol_parts[0] if len(symbol_parts) > 0 else parsed_symbol

            crypto = Cryptocurrency(
                symbol=parsed_symbol,
                name=name,
                is_active=True,
            )

            session.add(crypto)
            session.flush()  # Get ID without committing

            self._logger.info(f"Created new cryptocurrency record for {symbol}")

        return crypto

    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """
        Parse a symbol into exchange and base symbol components.

        Args:
            symbol: Trading symbol (e.g., "binance:BTC/USD")

        Returns:
            Tuple of (exchange, base_symbol)
        """
        if ":" in symbol:
            # Format: "exchange:base/quote"
            exchange, base_symbol = symbol.split(":", 1)
        else:
            # Format: "base/quote" (use default exchange)
            exchange = "default"
            base_symbol = symbol

        return exchange, base_symbol

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert a timeframe string to seconds.

        Args:
            timeframe: Timeframe string (e.g., "1h", "15m", "1d")

        Returns:
            Number of seconds in the timeframe
        """
        # Parse the number and unit from the timeframe
        if not timeframe:
            raise ValueError("Invalid timeframe: empty string")

        unit = timeframe[-1].lower()
        try:
            number = int(timeframe[:-1])
        except ValueError:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        # Convert to seconds
        if unit == "m":
            return number * 60
        elif unit == "h":
            return number * 60 * 60
        elif unit == "d":
            return number * 24 * 60 * 60
        elif unit == "w":
            return number * 7 * 24 * 60 * 60
        else:
            raise ValueError(f"Unknown timeframe unit: {unit}")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        include_indicators: bool = False,
        as_dataframe: bool = False,
    ) -> Union[List[OHLCV], pd.DataFrame]:
        """
        Get OHLCV data for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Timeframe of the data (e.g., "1h")
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of records to return
            include_indicators: Whether to include indicator values
            as_dataframe: Whether to return results as a pandas DataFrame

        Returns:
            List of OHLCV records or DataFrame

        Raises:
            DatabaseError: If there is an issue with the database operation
        """
        try:
            session = self.get_session()

            # Get cryptocurrency
            crypto = self._get_cryptocurrency(session, symbol)
            if not crypto:
                self._logger.warning(f"No cryptocurrency record found for {symbol}")
                return pd.DataFrame() if as_dataframe else []

            # Build query
            query = (
                session.query(OHLCV)
                .filter(OHLCV.cryptocurrency_id == crypto.id, OHLCV.timeframe == timeframe)
                .order_by(OHLCV.timestamp.asc())
            )

            if start_time:
                query = query.filter(OHLCV.timestamp >= start_time)
            if end_time:
                query = query.filter(OHLCV.timestamp <= end_time)
            if limit:
                query = query.limit(limit)

            # Execute query
            ohlcv_data = query.all()

            if not ohlcv_data:
                self._logger.warning(f"No OHLCV data found for {symbol} on {timeframe}")
                return pd.DataFrame() if as_dataframe else []

            # Convert to DataFrame if requested
            if as_dataframe:
                data = {
                    "timestamp": [record.timestamp for record in ohlcv_data],
                    "open": [float(record.open) for record in ohlcv_data],
                    "high": [float(record.high) for record in ohlcv_data],
                    "low": [float(record.low) for record in ohlcv_data],
                    "close": [float(record.close) for record in ohlcv_data],
                    "volume": [float(record.volume) for record in ohlcv_data],
                }

                if include_indicators:
                    indicator_keys = set()
                    # Find all indicator keys
                    for record in ohlcv_data:
                        if record.indicators:
                            indicator_keys.update(record.indicators.keys())

                    # Add indicator columns
                    for key in indicator_keys:
                        data[key] = [
                            (
                                float(record.indicators.get(key, np.nan))
                                if record.indicators
                                else np.nan
                            )
                            for record in ohlcv_data
                        ]

                df = pd.DataFrame(data)
                df.set_index("timestamp", inplace=True)
                return df

            return ohlcv_data

        except SQLAlchemyError as e:
            self._logger.exception(f"Database error retrieving OHLCV data for {symbol}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve OHLCV data: {str(e)}") from e
        except Exception as e:
            self._logger.exception(f"Unexpected error retrieving OHLCV data for {symbol}: {str(e)}")
            raise DataError(f"Failed to retrieve OHLCV data: {str(e)}") from e

    def get_market_snapshots(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        as_dataframe: bool = False,
    ) -> Union[List[MarketSnapshot], pd.DataFrame]:
        """
        Get market snapshots for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of records to return
            as_dataframe: Whether to return results as a pandas DataFrame

        Returns:
            List of MarketSnapshot records or DataFrame

        Raises:
            DatabaseError: If there is an issue with the database operation
        """
        try:
            session = self.get_session()

            # Get cryptocurrency
            crypto = self._get_cryptocurrency(session, symbol)
            if not crypto:
                self._logger.warning(f"No cryptocurrency record found for {symbol}")
                return pd.DataFrame() if as_dataframe else []

            # Build query
            query = (
                session.query(MarketSnapshot)
                .filter(MarketSnapshot.cryptocurrency_id == crypto.id)
                .order_by(MarketSnapshot.timestamp.desc())
            )

            if start_time:
                query = query.filter(MarketSnapshot.timestamp >= start_time)
            if end_time:
                query = query.filter(MarketSnapshot.timestamp <= end_time)
            if limit:
                query = query.limit(limit)

            # Execute query
            snapshots = query.all()

            if not snapshots:
                self._logger.warning(f"No market snapshots found for {symbol}")
                return pd.DataFrame() if as_dataframe else []

            # Convert to DataFrame if requested
            if as_dataframe:
                data = {
                    "timestamp": [snapshot.timestamp for snapshot in snapshots],
                    "id": [snapshot.id for snapshot in snapshots],
                }

                # Add ticker data if available
                ticker_fields = ["last", "bid", "ask", "volume"]
                for field in ticker_fields:
                    data[f"ticker_{field}"] = [
                        (
                            snapshot.data.get("ticker", {}).get(field, np.nan)
                            if snapshot.data and "ticker" in snapshot.data
                            else np.nan
                        )
                        for snapshot in snapshots
                    ]

                df = pd.DataFrame(data)
                df.set_index("timestamp", inplace=True)
                return df

            return snapshots

        except SQLAlchemyError as e:
            self._logger.exception(
                f"Database error retrieving market snapshots for {symbol}: {str(e)}"
            )
            raise DatabaseError(f"Failed to retrieve market snapshots: {str(e)}") from e
        except Exception as e:
            self._logger.exception(
                f"Unexpected error retrieving market snapshots for {symbol}: {str(e)}"
            )
            raise DataError(f"Failed to retrieve market snapshots: {str(e)}") from e
