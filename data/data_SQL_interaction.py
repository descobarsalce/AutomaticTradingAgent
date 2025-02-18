
"""SQL interaction handler with centralized database operations."""

from datetime import datetime, timedelta
import logging
import time
from typing import Dict, Optional, List
from sqlalchemy import and_, distinct, text
from sqlalchemy.exc import IntegrityError
from models.database import StockData
from utils.db_config import get_db_session
import pandas as pd

logger = logging.getLogger(__name__)

class SQLHandler:
    def __init__(self):
        self._session = None
        logger.info("📊 SQLHandler instance created")
        
    @property
    def session(self):
        if self._session is None or not self._session.is_active:
            self._session = next(get_db_session())
        return self._session

    def get_session(self, max_retries=3):
        """Get a database session with connection timeout"""
        start_time = datetime.now()
        connection_timeout = 10
        logger.info(f"Attempting database connection (timeout: {connection_timeout}s)")

        for retry_count in range(max_retries):
            try:
                if self.session is not None:
                    try:
                        self.session.execute(text("SELECT 1"))
                        logger.info("✅ Existing session verified")
                        return self.session
                    except Exception as e:
                        logger.warning(f"Existing session invalid: {str(e)}")
                        self.session.close()
                        self.session = None

                self.session = next(get_db_session())
                self.session.execute(text("SELECT 1"))
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ New database session established in {duration:.2f}s")
                return self.session

            except Exception as e:
                error_msg = f"Session creation attempt {retry_count + 1} failed: {str(e)}"
                logger.warning(error_msg)
                
                if self.session:
                    try:
                        self.session.close()
                    except Exception as close_error:
                        logger.warning(f"Error closing session: {str(close_error)}")
                    finally:
                        self.session = None

                if retry_count == max_retries - 1:
                    raise ConnectionError("Database connection failed after maximum retries") from e

                wait_time = min(2 ** retry_count, 10)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        logger.error("Session creation failed after all retries")
        return self.session

    def get_cached_data(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Retrieve cached data from database"""
        try:
            cached_records = self.session.query(StockData).filter(
                and_(
                    StockData.symbol == symbol,
                    StockData.date >= start_date,
                    StockData.date <= end_date
                )
            ).order_by(StockData.date).all()

            if not cached_records:
                return None

            data = pd.DataFrame([{
                'Close': record.close,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Volume': record.volume,
                'Date': record.date
            } for record in cached_records])

            data.set_index('Date', inplace=True)
            return data

        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache the fetched data in the database"""
        try:
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

                try:
                    self.session.add(stock_data)
                    self.session.flush()
                except IntegrityError:
                    self.session.rollback()
                    existing = self.session.query(StockData).filter(
                        StockData.symbol == symbol,
                        StockData.date == date
                    ).first()
                    if (existing.open != row['Open'] or
                        existing.high != row['High'] or
                        existing.low != row['Low'] or
                        existing.close != row['Close'] or
                        existing.volume != row['Volume']):
                        existing.open = row['Open']
                        existing.high = row['High']
                        existing.low = row['Low']
                        existing.close = row['Close']
                        existing.volume = row['Volume']
                        existing.last_updated = datetime.utcnow()

            self.session.commit()

        except Exception as e:
            logger.error(f"Error caching data: {e}")
            if self.session.is_active:
                self.session.rollback()
            raise

    def __del__(self):
        """Cleanup database session"""
        if self.session is not None:
            self.session.close()
