"""Data handler with data source abstraction and improved caching."""

import yfinance as yf
import pandas as pd
from time import sleep
from datetime import datetime, timedelta
from datetime import datetime
from data.data_SQL_interaction import SQLHandler
import logging
from typing import Dict, Optional, List, Protocol
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources."""
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass

import time

class YFinanceSource(DataSource):
    """YFinance implementation of data source with improved reliability."""
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        max_retries = 3
        base_delay = 2
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
        self._data_source = YFinanceSource()
        self._cache = {}
        logger.info("📈 DataHandler instance created")

    def execute_query(self, query):
        """Execute a database query through SQLHandler"""
        return self._sql_handler.session.execute(query)

    def query(self, *args, **kwargs):
        """Execute a database query through SQLHandler"""
        return self._sql_handler.session.query(*args, **kwargs)

    def fetch_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data with improved caching, validation and rate limiting."""
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
                # Add delay between requests to avoid rate limiting
                sleep(1)

                # Check cache first
                cached_data = self._sql_handler.get_cached_data(symbol, start_date, end_date)

                if cached_data is not None and not cached_data.empty:
                    stock_data = cached_data
                    logger.info(f"Using cached data for {symbol}")
                else:
                    logger.info(f"Fetching new data for {symbol}")
                    for attempt in range(3):  # Try up to 3 times
                        stock_data = self._data_source.fetch_data(symbol, start_date, end_date)
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