
"""Data handler with data source abstraction and improved caching."""

import yfinance as yf
import pandas as pd
from time import sleep
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, Optional, List, Protocol
import time
import os
from alpha_vantage.timeseries import TimeSeries

from data.base_sources import DataSource

logger = logging.getLogger(__name__)

class AlphaVantageSource(DataSource):
    """Alpha Vantage implementation of data source."""
    def __init__(self):
        import streamlit as st
        self.api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in secrets")
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

class DataHandler:
    def __init__(self):
        from data.data_SQL_interaction import SQLHandler
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
        """Fetch data using SQLHandler's data source hierarchy."""
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        elif isinstance(symbols, list):
            symbols = [s.strip() for s in symbols if s.strip()]

        if not symbols:
            raise ValueError("No symbols provided")

        result_df = pd.DataFrame()
        for symbol in symbols:
            try:
                data = self._sql_handler.get_data(symbol, start_date, end_date)
                if not data.empty:
                    data.columns = [f'{col}_{symbol}' for col in data.columns]
                    result_df = pd.concat([result_df, data], axis=1)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue

        if result_df.empty:
            raise ValueError("No data retrieved for any symbols")

        return result_df.copy()

    def __del__(self):
        """Cleanup database session"""
        if self._sql_handler:
            self._sql_handler._cleanup_session()
