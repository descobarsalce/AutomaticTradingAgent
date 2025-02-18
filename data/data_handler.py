"""
Data handler with centralized database configuration and improved error handling.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from data.data_feature_engineer import FeatureEngineer
from data.data_SQL_interaction import SQLHandler
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self):
        self._feature_engineer = None
        self._sql_handler = SQLHandler()
        self._cached_data = {}  # In-memory cache
        self._initialized = False
        logger.info("📈 DataHandler instance created")

    @property
    def feature_engineer(self):
        if self._feature_engineer is None:
            logger.info("🛠 Setting up FeatureEngineer...")
            self._feature_engineer = FeatureEngineer()
        return self._feature_engineer

    def prepare_data(self, data):
        """Prepare data using the feature engineer"""
        return self.feature_engineer.prepare_data(data)

    def prepare_data_with_extra_features(self, data):
        """Prepare data with extra features using the feature engineer"""
        prepared_data = self.feature_engineer.prepare_data(data)
        symbols = sorted(list(set(col.split('_')[1] for col in prepared_data.columns if '_' in col)))

        for symbol in symbols:
            close_col = f'Close_{symbol}'
            if close_col in prepared_data.columns:
                prepared_data = self.feature_engineer.add_lagged_features(prepared_data, [close_col], lags=5)
                prepared_data = self.feature_engineer.add_rolling_features(prepared_data, close_col, [7, 14, 30])
                fft_features = self.feature_engineer.add_fourier_transform(prepared_data[close_col])
                if not fft_features.empty:
                    for col in fft_features.columns:
                        prepared_data[f'{col}_{symbol}'] = fft_features[col]

        return prepared_data

    def _fetch_cached_data_if_valid(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Return valid cached data if present and complete, else None."""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        cached_data = self._sql_handler.get_cached_data(symbol, start_date, end_date)
        if cached_data is not None:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            if all(col in cached_data.columns for col in required_columns) \
                    and len(date_range.difference(cached_data.index)) == 0:
                self._cached_data[cache_key] = cached_data
                return cached_data
        return None

    def _fetch_data_yf(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        """Fetch data from yfinance and validate columns."""
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            if df.empty:
                logger.error(f"Empty dataset received for {symbol}")
                return pd.DataFrame()

            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Missing columns for {symbol}: {missing}")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_data(self, symbols, start_date, end_date):
        """Fetch data either from cache or yfinance, suffix columns by symbol."""
        if not self._initialized:
            self._initialized = True

        try:
            if isinstance(symbols, str):
                symbols = [symbols]

            if not symbols:
                raise ValueError("No symbols provided")

            if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
                raise ValueError("Invalid date format")

            if end_date <= start_date:
                raise ValueError("End date must be after start date")

            all_stocks_data = pd.DataFrame()

            for symbol in symbols:
                try:
                    cached_data = self._fetch_cached_data_if_valid(symbol, start_date, end_date)
                    if cached_data is not None:
                        stock_data = cached_data
                        logger.info(f"Using cached data for {symbol}")
                    else:
                        logger.info(f"Fetching new data for {symbol}")
                        stock_data = self._fetch_data_yf(symbol, start_date, end_date)
                        if not stock_data.empty:
                            self._sql_handler.cache_data(symbol, stock_data)
                        else:
                            logger.warning(f"Empty data received for {symbol}")
                            continue

                    stock_data.columns = [f'{col}_{symbol}' for col in stock_data.columns]
                    all_stocks_data = pd.concat([all_stocks_data, stock_data], axis=1)
                    logger.info(f"Successfully processed data for {symbol}")

                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    continue

            if all_stocks_data.empty:
                error_msg = "No valid data retrieved for any symbols. Check date range and symbol names."
                logger.error(error_msg)
                raise ValueError(error_msg)

            return all_stocks_data

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def __del__(self):
        """Cleanup database session"""
        if self._sql_handler:
            self._sql_handler.close_session()


    def get_cached_data(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Check if we have fresh data in the cache and validate continuity"""
        return None #This method is now handled by SQLHandler


    def cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache the fetched data in the database"""
        return None #This method is now handled by SQLHandler