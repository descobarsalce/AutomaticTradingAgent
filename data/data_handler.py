"""Data handler with data source abstraction and improved caching."""

import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Optional, List, Protocol
from sqlalchemy.exc import SQLAlchemyError

from data.data_SQL_interaction import SQLHandler
from data.stock_downloader import StockDownloader

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self):
        self._sql_handler = SQLHandler()
        self._cache = {}
        logger.info("📈 DataHandler instance created")

    def execute_query(self, query):
        """Execute a database query through SQLHandler"""
        return self._sql_handler.session.execute(query)

    def query(self, *args, **kwargs):
        """Execute a database query through SQLHandler"""
        return self._sql_handler.session.query(*args, **kwargs)

    
    def fetch_data(self, symbols: List[str], start_date: datetime, end_date: datetime, source="Alpha Vantage") -> pd.DataFrame:
        
        """Fetch data using source hierarchy: SQL -> Alpha Vantage -> YFinance"""
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        elif isinstance(symbols, list):
            symbols = [s.strip() for s in symbols if s.strip()]
        if not symbols:
            raise ValueError("No symbols provided")

        result_df = pd.DataFrame()
        for symbol in symbols:
            df = None
            
            # Try SQL first if not explicitly requesting other sources
            try:
                df = self._sql_handler.get_cached_data(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(f"Retrieved {symbol} data from SQL cache")
            except SQLAlchemyError as e:
                logger.warning(f"SQL error for {symbol}: {e}")

            # Use StockDownloader if SQL cache misses
            if df is None or df.empty:
                try:
                    downloader = StockDownloader(source='alpha_vantage' if source == "Alpha Vantage" else 'yahoo')
                    df = downloader.download_stock_data(symbol, start_date, end_date)
                    if not df.empty:
                        logger.info(f"Retrieved {symbol} data from {source}")
                        self._sql_handler.cache_data(symbol, df)
                except Exception as e:
                    logger.error(f"Download error for {symbol}: {str(e)}")

            if df is not None and not df.empty:
                df.columns = [f'{col}_{symbol}' for col in df.columns]
                result_df = pd.concat([result_df, df], axis=1)
            else:
                logger.error(f"Failed to fetch data for {symbol} from all sources")

        if result_df.empty:
            raise ValueError("No data retrieved for any symbols")

        return result_df.copy()

    def __del__(self):
        """Cleanup database session"""
        if hasattr(self, '_sql_handler'):
            self._sql_handler._cleanup_session()