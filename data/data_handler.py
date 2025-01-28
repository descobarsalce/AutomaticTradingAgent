"""
Data handler with centralized database configuration and improved error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.data_feature_engineer import FeatureEngineer
from utils.db_config import get_db_session
from sqlalchemy.orm import Session
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self):
        self.portfolio_data = {}
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
                cached_data = self._get_cached_data(symbol, start_date, end_date)
                if cached_data is not None and all(
                    col in cached_data.columns for col in required_columns
                ):
                    self.portfolio_data[symbol] = cached_data
                    continue

                # Only fetch from yfinance if cache miss or stale data
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if data.empty:
                    raise ValueError(f"No data retrieved for {symbol}")
                if not all(col in data.columns for col in required_columns):
                    raise ValueError(f"Missing required columns for {symbol}")

                self._cache_data(symbol, data)
                self.portfolio_data[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

        if not self.portfolio_data:
            raise ValueError("No valid data retrieved for any symbols")

        return self.portfolio_data

    def _get_cached_data(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Get cached data from database using centralized session"""
        try:
            # Implementation depends on your database schema
            # Use self.db_session to query the database
            pass # Placeholder - replace with actual DB query
        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None

    def _cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache data to database using centralized session"""
        try:
            # Implementation depends on your database schema
            # Use self.db_session to store the data
            pass # Placeholder - replace with actual DB insertion
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")

    def prepare_data(self):
        """Prepare data using feature engineer"""
        self.portfolio_data = self.feature_engineer.prepare_data(self.portfolio_data)
        return self.portfolio_data

    def __del__(self):
        """Cleanup database session"""
        if self._db_session is not None:
            self._db_session.close()