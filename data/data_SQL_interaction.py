"""SQL interaction handler with data source management and caching."""

import logging
import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from typing import Dict, Optional, List, Protocol, Tuple
from sqlalchemy import and_, func
from sqlalchemy.orm import Session
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.exc import IntegrityError

from data.database import StockData
from utils.db_config import get_db_session

class DataSource(ABC):
    """Abstract base class for data sources."""
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass

class AlphaVantageSource(DataSource):
    """Alpha Vantage implementation of data source."""
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')

    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            data, _ = self.ts.get_daily(symbol=symbol, outputsize='full')
            if data.empty:
                raise ValueError(f"Empty dataset received for {symbol}")

            # Rename columns to match format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Filter date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]

            if data.empty:
                raise ValueError(f"No data in specified date range for {symbol}")

            return data.round(2)

        except Exception as e:
            logger.warning(f"Alpha Vantage error for {symbol}: {str(e)}")
            return pd.DataFrame()

class YFinanceSource(DataSource):
    """YFinance implementation of data source with improved reliability."""
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        max_retries = 1
        base_delay = 1
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']

        for attempt in range(max_retries):
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    show_errors=False
                )

                if df.empty:
                    logger.warning(f"Empty dataset for {symbol} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (attempt + 1))
                        continue
                    return pd.DataFrame()

                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    logger.error(f"Missing columns for {symbol}: {missing}")
                    return pd.DataFrame()

                logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
                return df

            except Exception as e:
                logger.warning(f"YFinance error for {symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (attempt + 1))
                    continue
                return pd.DataFrame()


logger = logging.getLogger(__name__)

class SQLHandler:
    """Handles SQL database operations with improved caching and fallback sources."""

    BATCH_SIZE = 1000
    MAX_RETRIES = 3

    def __init__(self):
        self._session: Optional[Session] = None
        self._cache_status: Dict[Tuple[str, datetime, datetime], bool] = {}
        try:
            self._alpha_vantage = AlphaVantageSource()
            logger.info("📈 Alpha Vantage source initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Alpha Vantage: {e}")
            self._alpha_vantage = None
        self._yfinance = YFinanceSource()
        logger.info("📊 SQLHandler instance created with fallback sources")

    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Main method to fetch data with fallback hierarchy."""
        # Try SQL first
        df = self.get_cached_data(symbol, start_date, end_date)
        if df is not None and not df.empty:
            logger.info(f"Retrieved data from SQL cache for {symbol}")
            return df

        # Try Alpha Vantage
        if self._alpha_vantage:
            try:
                df = self._alpha_vantage.fetch_data(symbol, start_date, end_date)
                if not df.empty:
                    self.cache_data(symbol, df, start_date, end_date)
                    logger.info(f"Retrieved and cached Alpha Vantage data for {symbol}")
                    return df
            except Exception as e:
                logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")

        # Finally try YFinance
        try:
            df = self._yfinance.fetch_data(symbol, start_date, end_date)
            if not df.empty:
                self.cache_data(symbol, df, start_date, end_date)
                logger.info(f"Retrieved and cached YFinance data for {symbol}")
                return df
        except Exception as e:
            logger.error(f"YFinance fetch failed for {symbol}: {e}")

        return pd.DataFrame()

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
            if self.session.in_transaction():
                self.session.rollback()

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

            expected_dates = pd.date_range(start=start_date, end=end_date, freq='B')
            if len(df) < len(expected_dates) * 0.9:
                logger.warning(f"Incomplete cached data for {symbol}")
                return None

            return df

        except SQLAlchemyError as e:
            if self.session.in_transaction():
                self.session.rollback()
            logger.error(f"Database error: {str(e)}")
            return None

    def cache_data(self, symbol: str, data: pd.DataFrame, start_date: datetime, end_date: datetime) -> None:
        """Cache data with improved batching and validation."""
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

            for i in range(0, len(records), self.BATCH_SIZE):
                batch = records[i:i + self.BATCH_SIZE]
                self._process_batch(batch)

            cache_key = self._get_cache_key(symbol, start_date, end_date)
            self._cache_status[cache_key] = True

        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            if self.session.is_active:
                self.session.rollback()
            raise

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True
    )
    def _process_batch(self, batch: list) -> None:
        """Process a batch of records with conflict resolution."""
        for record in batch:
            try:
                stock_data = StockData(**record)
                self.session.add(stock_data)
                self.session.flush()
            except IntegrityError:
                self.session.rollback()
                self._update_existing_record(record)

        self.session.commit()

    def _update_existing_record(self, record: Dict) -> None:
        """Update existing record with new data."""
        existing = self.session.query(StockData).filter(
            StockData.symbol == record['symbol'],
            StockData.date == record['date']
        ).first()

        if existing:
            existing.open = record['open']
            existing.high = record['high']
            existing.low = record['low']
            existing.close = record['close']
            existing.volume = record['volume']
            existing.last_updated = record['last_updated']

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

from data.alpha_vantage_source import AlphaVantageSource