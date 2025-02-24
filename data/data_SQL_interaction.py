"""SQL interaction handler with data source management and caching."""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple
from sqlalchemy import and_, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from tenacity import retry, stop_after_attempt, wait_exponential

from data.database import StockData
from utils.db_config import get_db_session

logger = logging.getLogger(__name__)

class SQLHandler:
    """Handles SQL database operations with improved caching."""

    BATCH_SIZE = 1000
    MAX_RETRIES = 3

    def __init__(self):
        self._session: Optional[Session] = None
        self._cache_status: Dict[Tuple[str, datetime, datetime], bool] = {}
        logger.info("📊 SQLHandler instance created")

    @property
    def session(self) -> Session:
        """Lazily load a database session."""
        if self._session is None or not self._session.is_active:
            self._session = next(get_db_session())
        return self._session

    def _get_cache_key(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[str, datetime, datetime]:
        return (symbol, start_date, end_date)

    def is_data_cached(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """Check if data is fully cached for the given symbol and date range."""
        cache_key = self._get_cache_key(symbol, start_date, end_date)

        if cache_key in self._cache_status:
            return self._cache_status[cache_key]

        count = self.session.query(func.count(StockData.id)).filter(
            and_(
                StockData.symbol == symbol,
                StockData.date >= start_date,
                StockData.date <= end_date
            )
        ).scalar()

        expected_days = len(pd.date_range(start=start_date, end=end_date, freq='B'))
        is_cached = (count == expected_days)
        self._cache_status[cache_key] = is_cached

        return is_cached

    def get_cached_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Retrieve cached data with validation."""
        try:
            self.session.rollback()  # Clear any failed transaction state
            query = self.session.query(StockData).filter(
                and_(
                    StockData.symbol == symbol,
                    StockData.date >= start_date,
                    StockData.date <= end_date
                )
            ).order_by(StockData.date)

            records = query.all()
            if not records:
                logger.info(f"No cached data found for {symbol}")
                return None

            df = pd.DataFrame([{
                'Close': record.close,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Volume': record.volume,
                'Date': record.date
            } for record in records]).set_index('Date')

            return df

        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=1, max=8))
    def cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache data with improved batching and validation."""
        try:
            self.session.rollback()  # Clear any failed transaction state
            records = []
            for date, row in data.iterrows():
                stock_data = StockData(
                    symbol=symbol,
                    date=date,
                    close=row['Close'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    volume=row['Volume'],
                    last_updated=datetime.utcnow()
                )
                records.append(stock_data)

            for i in range(0, len(records), self.BATCH_SIZE):
                batch = records[i:i + self.BATCH_SIZE]
                self._process_batch(batch)

        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            raise

    def _process_batch(self, batch: list) -> None:
        """Process a batch of records with conflict resolution."""
        for record in batch:
            try:
                self.session.merge(record)
            except Exception as e:
                logger.error(f"Error processing record: {str(e)}")
                self.session.rollback()
                raise

        self.session.commit()

    def _cleanup_session(self):
        """Close the session if open."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {str(e)}")
            finally:
                self._session = None

    def __del__(self):
        """Ensure session cleanup on deletion."""
        self._cleanup_session()