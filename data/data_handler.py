"""
Data handler with centralized database configuration and improved error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
# from datetime import datetime, timedelta
from data.data_feature_engineer import FeatureEngineer
from utils.db_config import get_db_session
from sqlalchemy.orm import Session
import logging
from typing import Dict, Optional
from models.database import Session, StockData

logger = logging.getLogger(__name__)


class DataHandler:

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self._db_session: Optional[Session] = None

    @property
    def db_session(self) -> Session:
        """Lazy database session initialization"""
        if self._db_session is None:
            self._db_session = get_db_session()
        return self._db_session

    def fetch_data(self, symbols, start_date, end_date):
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if isinstance(symbols, str):
            symbols = [symbols]

        for symbol in symbols:
            try:
                # Use the database session from centralized config
                cached_data = self._get_cached_data(symbol, start_date,
                                                    end_date)
                if cached_data is not None and all(
                        col in cached_data.columns
                        for col in required_columns):
                    portfolio_data[symbol] = cached_data
                    continue

                # Only fetch from yfinance if cache miss or stale data
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if data.empty:
                    raise ValueError(f"No data retrieved for {symbol}")
                if not all(col in data.columns for col in required_columns):
                    raise ValueError(f"Missing required columns for {symbol}")

                self.cache_data(symbol, data)
                portfolio_data[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

        if not portfolio_data:
            raise ValueError("No valid data retrieved for any symbols")

        return portfolio_data

    def prepare_data(
            self,
            portfolio_data: Dict[str,
                                 pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data using feature engineer"""
        return self.feature_engineer.prepare_data(portfolio_data)

    def __del__(self):
        """Cleanup database session"""
        if self._db_session is not None:
            self._db_session.close()

    def get_cached_data(self, symbol: str, start_date, end_date):
        """Check if we have fresh data in the cache"""
        try:
            # Query cached data
            cached_records = self.session.query(StockData).filter(
                and_(StockData.symbol == symbol, StockData.date >= start_date,
                     StockData.date
                     <= end_date)).order_by(StockData.date).all()

            if not cached_records:
                return None

            # Check if data is fresh (less than 1 day old)
            newest_record = max(record.last_updated
                                for record in cached_records)
            if datetime.utcnow() - newest_record > timedelta(days=1):
                return None

            # Convert to DataFrame
            data = pd.DataFrame([{
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Close': record.close,
                'Volume': record.volume,
                'Date': record.date
            } for record in cached_records])

            data.set_index('Date', inplace=True)
            return data

        except Exception as e:
            print(f"Error reading from cache: {e}")
            return None

    def cache_data(self, symbol: str, data: pd.DataFrame):
        """Cache the fetched data in the database"""
        try:
            for date, row in data.iterrows():
                stock_data = StockData(symbol=symbol,
                                       date=date,
                                       open=row['Open'],
                                       high=row['High'],
                                       low=row['Low'],
                                       close=row['Close'],
                                       volume=row['Volume'],
                                       last_updated=datetime.utcnow())
                # Handle conflict resolution by using merge with unique constraint
                existing = self.session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.date == date).first()

                if existing:
                    # Update existing record
                    existing.open = row['Open']
                    existing.high = row['High']
                    existing.low = row['Low']
                    existing.close = row['Close']
                    existing.volume = row['Volume']
                    existing.last_updated = datetime.utcnow()
                else:
                    # Insert new record
                    self.session.add(stock_data)

            self.session.commit()

        except Exception as e:
            print(f"Error caching data: {e}")
            self.session.rollback()
