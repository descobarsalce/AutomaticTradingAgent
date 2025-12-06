"""Data handler with data source abstraction and improved caching."""

from datetime import datetime
import logging
from typing import List, Dict, Optional, Union

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from data.data_SQL_interaction import SQLHandler
from data.stock_downloader import StockDownloader

logger = logging.getLogger(__name__)

REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def ensure_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert a datetime-like object to a timezone-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a timezone-aware DatetimeIndex sorted by time."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index()
    if frame.index.isna().any():
        raise ValueError("DataFrame index contains NaT values")
    return frame


def validate_ohlcv_frame(frame: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Validate OHLCV schema, non-null columns, and timezone-aware index."""
    if frame is None or frame.empty:
        raise ValueError("Empty data provided")

    frame = ensure_datetime_index(frame)

    for symbol in symbols:
        for col in REQUIRED_OHLCV_COLUMNS:
            col_name = f"{col}_{symbol}"
            if col_name not in frame.columns:
                raise ValueError(f"Missing column {col_name}")
            if frame[col_name].isna().any():
                raise ValueError(f"Column {col_name} contains null values")

    return frame


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

    
    def _validate_and_parse_symbols(self, symbols: Union[str, List[str]]) -> List[str]:
        """Validate and parse input symbols."""
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        elif isinstance(symbols, list):
            symbols = [s.strip() for s in symbols if s.strip()]
        if not symbols:
            raise ValueError("No symbols provided")
        return symbols

    def _fetch_from_sql(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Attempt to fetch data from SQL cache."""
        try:
            df = self._sql_handler.get_cached_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                logger.info(f"Retrieved {symbol} data from SQL cache")
                return df
        except SQLAlchemyError as e:
            logger.warning(f"SQL error for {symbol}: {e}")
        return None

    def _fetch_from_external(self, symbol: str, start_date: datetime, end_date: datetime, source: str) -> Optional[pd.DataFrame]:
        """Fetch data from external source and cache it."""
        try:
            logger.info(f"Fetching {symbol} from {source}: [{start_date}, {end_date}]")
            downloader = StockDownloader(source='alpha_vantage' if source == "Alpha Vantage" else 'yahoo')
            df = downloader.download_stock_data(symbol, start_date, end_date)

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} from {source}")
                return None

            logger.info(f"Retrieved {len(df)} rows for {symbol} from {source}")

            try:
                self._sql_handler.cache_data(symbol, df)
                logger.info(f"Cached {symbol} data in SQL")
            except Exception as cache_error:
                logger.warning(f"Failed to cache {symbol}: {str(cache_error)}")
                # Don't fail the entire operation if caching fails

            return df

        except Exception as e:
            logger.error(f"Download error for {symbol} from {source}: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    def _process_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process dataframe by adding symbol suffix to columns except Date."""
        if df is not None and not df.empty:
            # First set the Date as index if it's a column
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'Date'})
            
            # Add symbol suffix to all columns except Date
            cols_to_rename = [col for col in df.columns if col != 'Date']
            df.columns = ['Date' if col == 'Date' else f'{col}_{symbol}' for col in df.columns]
            
            # Set Date as index if not already
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            
        return df

    def fetch_data(self, symbols: Union[str, List[str]], start_date: datetime, end_date: datetime, source="Alpha Vantage", use_SQL=True) -> pd.DataFrame:
        """Fetch data with improved error handling and validation.

        Args:
            symbols: List of stock symbols or comma-separated string
            start_date: Start date for data fetching
            end_date: End date for data fetching
            source: Data source ("Alpha Vantage" or "Yahoo")
            use_SQL: Whether to use SQL cache first

        Returns:
            DataFrame containing the fetched data
        """
        # INPUT VALIDATION
        start_ts = ensure_utc_timestamp(start_date)
        end_ts = ensure_utc_timestamp(end_date)

        if start_ts > end_ts:
            raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date})")

        logger.info(f"Fetching data for {symbols} from {source}: [{start_date}, {end_date}]")

        symbols = self._validate_and_parse_symbols(symbols)
        result_df = pd.DataFrame()

        for symbol in symbols:
            df = None
            if use_SQL:
                df = self._fetch_from_sql(symbol, start_ts.to_pydatetime(), end_ts.to_pydatetime())
                logger.info(f"Using SQL cache for: {symbol}")
                if df is not None:
                    logger.info(f"Retrieved df columns: {list(df.columns)}")
                    logger.info(f"Retrieved df shape: {df.shape}")
                    # SQL data already has symbol suffix, just need to set Date as index
                    if 'Date' in df.columns and df.index.name != 'Date':
                        df.set_index('Date', inplace=True)
                        logger.info(f"Set Date as index for {symbol}")

            if df is None or not use_SQL:
                df = self._fetch_from_external(symbol, start_ts.to_pydatetime(), end_ts.to_pydatetime(), source)
                if df is not None:
                    if not df.empty:
                        df = self._process_dataframe(df, symbol)
            
            if df is not None:
                result_df = pd.concat([result_df, df], axis=1)
                logger.info(f"Appended {symbol} data to result_df")
            else:
                logger.error(f"Failed to fetch data for {symbol} from all sources")

        if result_df is None or result_df.empty:
            logger.info(f"No data retrieved for any symbols")
            raise ValueError("No data retrieved for any symbols")

        result_df = validate_ohlcv_frame(result_df, symbols)

        return result_df.copy()
    
    def __del__(self):
        """Cleanup database session"""
        if hasattr(self, '_sql_handler'):
            self._sql_handler._cleanup_session()