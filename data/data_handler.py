"""Data handler with data source abstraction and improved caching."""

import yfinance as yf
import pandas as pd
from time import sleep
from datetime import datetime, timedelta, timezone
from data.data_SQL_interaction import SQLHandler
import logging
from typing import Dict, Optional, List, Protocol
from abc import ABC, abstractmethod
import time
import os
from alpha_vantage.timeseries import TimeSeries
import asyncio

logger = logging.getLogger(__name__)

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

    async def fetch_intraday_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            data, _ = self.ts.get_intraday(symbol=symbol, interval='60min', outputsize='full')
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
                # Force download to bypass cache
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

                # Ensure all required columns exist
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    logger.error(f"Missing columns for {symbol}: {missing}")
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

class DataHandler:
    def __init__(self):
        self._sql_handler = SQLHandler()
        try:
            self._data_source = AlphaVantageSource()
            logger.info("📈 Alpha Vantage source initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Alpha Vantage: {e}")
            self._data_source = None
        self._fallback_source = YFinanceSource()
        self._cache = {}
        logger.info("📈 DataHandler instance created")

    def execute_query(self, query):
        """Execute a database query through SQLHandler"""
        return self._sql_handler.session.execute(query)

    def query(self, *args, **kwargs):
        """Execute a database query through SQLHandler"""
        return self._sql_handler.session.query(*args, **kwargs)

    def fetch_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data with simple processing and caching."""
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        elif isinstance(symbols, list):
            symbols = [s.strip() for s in symbols if s.strip()]

        if not symbols:
            raise ValueError("No symbols provided")

        result_df = pd.DataFrame()
        for symbol in symbols:
            try:
                if self._data_source:
                    data = await self._data_source.fetch_intraday_data(symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        # Rename columns with symbol suffix
                        data.columns = [f'{col}_{symbol}' for col in data.columns]
                        result_df = pd.concat([result_df, data], axis=1)
                    else:
                        # Fallback to YFinance
                        data = self._fallback_source.fetch_data(symbol, start_date, end_date)
                        if not data.empty:
                            data.columns = [f'{col}_{symbol}' for col in data.columns]
                            result_df = pd.concat([result_df, data], axis=1)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue

        if result_df.empty:
            raise ValueError("No data retrieved for any symbols")

        return result_df.copy()
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        elif isinstance(symbols, list):
            symbols = [s.strip() for s in symbols if s.strip()]

        if not symbols:
            raise ValueError("No symbols provided")

        # Validate symbols
        if not all(isinstance(s, str) and s for s in symbols):
            raise ValueError("Invalid symbol format")

        all_stocks_data = pd.DataFrame()
        failed_symbols = []

        for symbol in symbols:
            try:
                # Check cache first
                cached_data = self._sql_handler.get_cached_data(symbol, start_date, end_date)

                if cached_data is not None and not cached_data.empty:
                    stock_data = cached_data
                    logger.info(f"Using cached data for {symbol}")
                else:
                    logger.info(f"Fetching new data for {symbol}")
                    if self._data_source:
                        stock_data = await self._data_source.fetch_intraday_data(symbol, start_date, end_date)
                    if stock_data.empty and self._fallback_source:
                        logger.warning(f"Falling back to YFinance for {symbol}")
                        for attempt in range(3):  # Try up to 3 times
                            stock_data = self._fallback_source.fetch_data(symbol, start_date, end_date)
                            if not stock_data.empty:
                                break
                            sleep(2 * (attempt + 1))  # Exponential backoff

                    if not stock_data.empty:
                        self._sql_handler.cache_data(symbol, stock_data, start_date, end_date)
                    else:
                        logger.warning(f"Empty data received for {symbol} after retries")
                        failed_symbols.append(symbol)
                        continue

                # Validate data
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in stock_data.columns for col in required_cols):
                    logger.error(f"Missing required columns for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                # Suffix columns with symbol
                stock_data.columns = [f'{col}_{symbol}' for col in stock_data.columns]
                all_stocks_data = pd.concat([all_stocks_data, stock_data], axis=1)
                logger.info(f"Successfully processed data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue

        if all_stocks_data.empty:
            raise ValueError(f"No valid data retrieved for any symbols. Failed symbols: {failed_symbols}")
        elif failed_symbols:
            logger.warning(f"Data retrieval failed for symbols: {failed_symbols}")

        return all_stocks_data

    def __del__(self):
        """Cleanup database session"""
        if self._sql_handler:
            self._sql_handler._cleanup_session()