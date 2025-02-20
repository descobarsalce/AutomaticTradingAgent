"""Data handler with data source abstraction and improved caching."""

import yfinance as yf
import pandas as pd
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

class YFinanceSource(DataSource):
    """YFinance implementation of data source."""
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            if df.empty:
                logger.error(f"Empty dataset received for {symbol}")
                return pd.DataFrame()

            required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Missing columns for {symbol}: {missing}")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {str(e)}")
            return pd.DataFrame()

class DataHandler:
    def __init__(self):
        self._sql_handler = SQLHandler()
        self._data_source = YFinanceSource()
        self._cache = {}
        logger.info("📈 DataHandler instance created")

    def fetch_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data with improved caching and validation."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if not symbols:
            raise ValueError("No symbols provided")

        all_stocks_data = pd.DataFrame()

        for symbol in symbols:
            try:
                # Check cache first
                cached_data = self._sql_handler.get_cached_data(symbol, start_date, end_date)

                if cached_data is not None:
                    stock_data = cached_data
                    logger.info(f"Using cached data for {symbol}")
                else:
                    logger.info(f"Fetching new data for {symbol}")
                    stock_data = self._data_source.fetch_data(symbol, start_date, end_date)

                    if not stock_data.empty:
                        self._sql_handler.cache_data(symbol, stock_data, start_date, end_date)
                    else:
                        logger.warning(f"Empty data received for {symbol}")
                        continue

                # Suffix columns with symbol
                stock_data.columns = [f'{col}_{symbol}' for col in stock_data.columns]
                all_stocks_data = pd.concat([all_stocks_data, stock_data], axis=1)
                logger.info(f"Successfully processed data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue

        if all_stocks_data.empty:
            raise ValueError("No valid data retrieved for any symbols")

        return all_stocks_data

    def __del__(self):
        """Cleanup database session"""
        if self._sql_handler:
            self._sql_handler._cleanup_session()