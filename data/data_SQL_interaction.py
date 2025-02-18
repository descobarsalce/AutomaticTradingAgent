
"""SQL interaction handler with improved structure and performance optimizations."""

from datetime import datetime, timedelta
import logging
import time
from typing import Dict, Optional, List, Any
from sqlalchemy import and_, distinct, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from models.database import StockData
from utils.db_config import get_db_session
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class SQLHandler:
    """Handles SQL database operations with connection pooling and retries."""
    
    BATCH_SIZE = 1000
    MAX_RETRIES = 3
    
    def __init__(self):
        self._session = None
        logger.info("📊 SQLHandler instance created")
    
    @property
    def session(self) -> Session:
        """Get or create a database session."""
        if self._session is None or not self._session.is_active:
            self._session = next(get_db_session())
        return self._session

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _execute_with_retry(self, operation: callable, *args, **kwargs) -> Any:
        """Execute database operation with retry logic."""
        try:
            return operation(*args, **kwargs)
        except SQLAlchemyError as e:
            logger.error(f"Database operation failed: {str(e)}")
            self._cleanup_session()
            raise

    def _cleanup_session(self) -> None:
        """Clean up the current session."""
        if self._session:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {str(e)}")
            finally:
                self._session = None

    def get_cached_data(self, symbol: str, start_date: datetime, 
                       end_date: datetime) -> Optional[pd.DataFrame]:
        """Retrieve cached data from database with error handling."""
        try:
            query = self.session.query(StockData).filter(
                and_(
                    StockData.symbol == symbol,
                    StockData.date >= start_date,
                    StockData.date <= end_date
                )
            ).order_by(StockData.date)
            
            records = self._execute_with_retry(query.all)
            
            if not records:
                return None

            data = pd.DataFrame([{
                'Close': record.close,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Volume': record.volume,
                'Date': record.date
            } for record in records])

            data.set_index('Date', inplace=True)
            return data

        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None

    def cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache data in batches with optimized performance."""
        try:
            records = []
            for date, row in data.iterrows():
                stock_data = {
                    'symbol': symbol,
                    'date': date,
                    'close': row['Close'],
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'volume': row['Volume'],
                    'last_updated': datetime.utcnow()
                }
                records.append(stock_data)

            # Process in batches
            for i in range(0, len(records), self.BATCH_SIZE):
                batch = records[i:i + self.BATCH_SIZE]
                self._process_batch(batch)

        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            if self.session.is_active:
                self.session.rollback()
            raise

    def _process_batch(self, batch: List[Dict]) -> None:
        """Process a batch of records with conflict resolution."""
        for record in batch:
            try:
                stock_data = StockData(**record)
                self.session.add(stock_data)
                self.session.flush()
            except IntegrityError:
                self.session.rollback()
                existing = self.session.query(StockData).filter(
                    StockData.symbol == record['symbol'],
                    StockData.date == record['date']
                ).first()
                
                if self._should_update_record(existing, record):
                    self._update_existing_record(existing, record)
            
        self.session.commit()

    def _should_update_record(self, existing: StockData, new_data: Dict) -> bool:
        """Determine if an existing record should be updated."""
        return (existing.open != new_data['open'] or
                existing.high != new_data['high'] or
                existing.low != new_data['low'] or
                existing.close != new_data['close'] or
                existing.volume != new_data['volume'])

    def _update_existing_record(self, existing: StockData, new_data: Dict) -> None:
        """Update an existing record with new data."""
        existing.open = new_data['open']
        existing.high = new_data['high']
        existing.low = new_data['low']
        existing.close = new_data['close']
        existing.volume = new_data['volume']
        existing.last_updated = new_data['last_updated']

    def __del__(self):
        """Cleanup database session on object destruction."""
        self._cleanup_session()
