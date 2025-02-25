
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
from sqlalchemy import and_, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from tenacity import retry, stop_after_attempt, wait_exponential

from data.database import StockData  # <-- Your ORM model
from utils.db_config import get_db_session

logger = logging.getLogger(__name__)

class SQLHandler:
    """
    Handles SQL database operations with improved caching, robust session handling,
    and more explicit error management.
    """

    BATCH_SIZE = 1000
    MAX_RETRIES = 3

    def __init__(self):
        self._session: Optional[Session] = None
        # Cache key: (symbol, start_date, end_date), Value: boolean indicating if data is fully cached
        self._cache_status: Dict[Tuple[str, datetime, datetime], bool] = {}
        logger.info("📊 SQLHandler instance created")

    @property
    def session(self) -> Session:
        """
        Lazily loads a database session with proper transaction handling.
        Using a property for session can be risky if concurrency is needed.
        Evaluate whether a session-per-request or context-manager pattern
        is more suitable for your environment.
        """
        if self._session is None or not self._session.is_active:
            self._cleanup_session(force=True)  # Make sure we close any stale session first
            self._session = next(get_db_session())
        return self._session

    def _cleanup_session(self, force: bool = False):
        """
        Safely close and remove the current session from memory.
        If `force` is True, forcibly closes the session even if there's no error.
        """
        if self._session:
            try:
                self._session.rollback()
                self._session.close()
            except Exception as e:
                logger.warning(f"Error rolling back or closing session: {str(e)}")
            finally:
                if force:
                    self._session = None

    def __del__(self):
        """
        Ensure session cleanup on deletion.
        Note: Relying on __del__ can be unreliable in certain Python implementations.
        """
        self._cleanup_session(force=True)

    def _get_cache_key(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Tuple[str, datetime, datetime]:
        return (symbol, start_date, end_date)

    def is_data_cached(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> bool:
        """
        Check if data is fully cached for the given symbol and date range.
        This method uses a simple day count comparison (expected_days vs. row count).
        If your data may have missing market days or partial data, consider a more
        flexible check or store that logic separately.
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date)

        if cache_key in self._cache_status:
            return self._cache_status[cache_key]

        try:
            # Rely on active session
            query = (
                self.session.query(func.count(StockData.id))
                .filter(
                    and_(
                        StockData.symbol == symbol,
                        StockData.date >= start_date,
                        StockData.date <= end_date,
                    )
                )
            )
            row_count = query.scalar()
        except Exception as e:
            logger.error(f"Error querying data count for cache check: {str(e)}")
            row_count = 0

        # Calculate expected number of business days between start_date and end_date.
        expected_days = len(pd.date_range(start=start_date, end=end_date, freq="B"))

        # Mark as cached only if the row count matches expected business days
        is_cached = (row_count == expected_days)
        self._cache_status[cache_key] = is_cached

        return is_cached

    def get_cached_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data with validation.
        Returns a pandas DataFrame with the data indexed by date if found.
        Returns None if no data is present or if an error occurred.
        """
        try:
            # Start fresh with a new/clean session to avoid stale states
            self._cleanup_session(force=True)
            local_session = next(get_db_session())

            query = (
                local_session.query(StockData)
                .filter(
                    and_(
                        StockData.symbol == symbol,
                        StockData.date >= start_date,
                        StockData.date <= end_date,
                    )
                )
                .order_by(StockData.date)
            )
            records = query.all()
            local_session.close()

            if not records:
                logger.info(f"No cached data found for {symbol} in [{start_date}, {end_date}]")
                return None

            df = pd.DataFrame(
                [
                    {
                        "Date": record.date,
                        "Close": record.close,
                        "Open": record.open,
                        "High": record.high,
                        "Low": record.low,
                        "Volume": record.volume,
                    }
                    for record in records
                ]
            ).set_index("Date")

            return df

        except Exception as e:
            logger.error(f"Database error while retrieving cached data: {str(e)}")
            return None

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Cache data with improved batching and validation.
        If the data is large, this method breaks inserts/updates into chunks.
        Uses the `tenacity` library to retry on transient failures.
        """
        if data.empty:
            logger.warning(f"No data provided to cache for symbol '{symbol}'.")
            return

        # Make sure DataFrame index is DateTime-like
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DateTimeIndex.")

        try:
            # Clear any failed transaction state in the existing session
            self.session.rollback()

            # Convert each row into a StockData object
            records = []
            for date, row in data.iterrows():
                stock_data = StockData(
                    symbol=symbol,
                    date=date,
                    close=row["Close"],
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    volume=row["Volume"],
                    last_updated=datetime.utcnow(),
                )
                records.append(stock_data)

            # Insert or update records in batches
            for i in range(0, len(records), self.BATCH_SIZE):
                batch = records[i : i + self.BATCH_SIZE]
                self._process_batch(batch)

        except IntegrityError as ie:
            logger.error(f"Integrity error while caching data: {str(ie)}")
            self.session.rollback()
            raise

        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            self.session.rollback()
            raise

    def _process_batch(self, batch: list) -> None:
        """
        Process a batch of records with conflict resolution using `session.merge()`.
        Commits the transaction once per batch.
        """
        try:
            for record in batch:
                self.session.merge(record)
            self.session.commit()
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            self.session.rollback()
            raise

    def reset_cache_status(self) -> None:
        """
        Clears the in-memory cache status. Useful when you know new data
        may invalidate the old cache checks. 
        """
        self._cache_status.clear()

    def refresh_session(self) -> None:
        """
        Manually reset the session if desired.
        """
        self._cleanup_session(force=True)
        logger.info("Session has been refreshed.")
