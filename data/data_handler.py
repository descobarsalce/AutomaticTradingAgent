"""
Data handler with centralized database configuration and improved error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import and_, distinct
from datetime import timedelta, datetime
from data.data_feature_engineer import FeatureEngineer
from utils.db_config import get_db_session
import logging
from typing import Dict, Optional
from models.database import StockData

logger = logging.getLogger(__name__)

class DataHandler:

    def __init__(self):
        logger.info("📈 Initializing DataHandler...")
        start_time = datetime.now()
        
        logger.info("🛠 Setting up FeatureEngineer...")
        self.feature_engineer = FeatureEngineer()
        self.session = None
        
        logger.info(f"✅ DataHandler initialization completed in {(datetime.now() - start_time).total_seconds():.2f}s")

    def get_session(self, max_retries=3):
        start_time = datetime.now()
        logger.info(f"🔌 Attempting to establish database session (max retries: {max_retries})...")
        """Get a database session using context management with retries"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.debug(f"Attempt {retry_count + 1} of {max_retries}")
                if self.session is None or not self.session.is_active:
                    logger.debug("Creating new database session")
                    self.session = next(get_db_session())
                    logger.debug(f"Session created: {self.session is not None}")
                
                if self.session and self.session.is_active:
                    logger.info(f"✅ Database session established successfully in {(datetime.now() - start_time).total_seconds():.2f}s")
                    return self.session
                else:
                    logger.warning("Session created but not active")
                    
            except Exception as e:
                logger.warning(f"Session creation attempt {retry_count + 1} failed: {str(e)}")
                logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
                if self.session:
                    self.session.close()
                self.session = None
                retry_count += 1
                if retry_count == max_retries:
                    logger.error("Failed to create database session after maximum retries")
                    raise
        
        logger.error("Session creation failed after all retries")
        return self.session

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
                # Add lagged features
                prepared_data = self.feature_engineer.add_lagged_features(prepared_data, [close_col], lags=5)
                # Add rolling features
                prepared_data = self.feature_engineer.add_rolling_features(prepared_data, close_col, [7, 14, 30])
                # Add Fourier transform features
                fft_features = self.feature_engineer.add_fourier_transform(prepared_data[close_col])
                if not fft_features.empty:
                    for col in fft_features.columns:
                        prepared_data[f'{col}_{symbol}'] = fft_features[col]

        return prepared_data

    def _fetch_cached_data_if_valid(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Return valid cached data if present and complete, else None."""
        cached_data = self.get_cached_data(symbol, start_date, end_date)
        if cached_data is not None:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            if all(col in cached_data.columns for col in required_columns) \
                    and len(date_range.difference(cached_data.index)) == 0:
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
                        self.cache_data(symbol, stock_data)
                    else:
                        logger.warning(f"Empty data received for {symbol}")
                        continue

                # Rename columns to include symbol
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

    def __del__(self):
        """Cleanup database session"""
        if self.session is not None:
            self.session.close()

    def get_cached_data(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Check if we have fresh data in the cache and validate continuity"""
        try:
            cached_records = self.session.query(StockData).filter(
                and_(
                    StockData.symbol == symbol,
                    StockData.date >= start_date,
                    StockData.date <= end_date
                )
            ).order_by(StockData.date).all()

            if not cached_records:
                return None

            # Check data continuity
            dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
            available_dates = set(record.date.date() for record in cached_records)
            missing_dates = [d for d in dates if d.date() not in available_dates]
            
            if len(missing_dates) > 0:
                logger.warning(f"Missing {len(missing_dates)} trading days for {symbol}")
                return None  # Force refetch if data is incomplete
                
            # Only check staleness for the most recent dates
            if end_date >= (datetime.now(datetime.timezone.utc) - timedelta(days=5)).date():
                newest_record = max(record.last_updated for record in cached_records)
                if datetime.now(datetime.timezone.utc) - newest_record > timedelta(days=1):
                    return None

            data = pd.DataFrame([{
                'Close': record.close,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Volume': record.volume,
                'Date': record.date
            } for record in cached_records])

            data.set_index('Date', inplace=True)
            return data

        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache the fetched data in the database"""
        try:
            for date, row in data.iterrows():
                stock_data = StockData(
                    symbol=symbol,
                    date=date,
                    close=row['Close'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    volume=row['Volume'],
                    last_updated=datetime.utcnow()
                )

                try:
                    self.session.add(stock_data)
                    self.session.flush()
                except IntegrityError:
                    self.session.rollback()
                    # Only update if data values are different
                    existing = self.session.query(StockData).filter(
                        StockData.symbol == symbol,
                        StockData.date == date
                    ).first()
                    if (existing.open != row['Open'] or
                        existing.high != row['High'] or
                        existing.low != row['Low'] or
                        existing.close != row['Close'] or
                        existing.volume != row['Volume']):
                        existing.open = row['Open']
                        existing.high = row['High']
                        existing.low = row['Low']
                        existing.close = row['Close']
                        existing.volume = row['Volume']
                        existing.last_updated = datetime.utcnow()

            self.session.commit()

        except Exception as e:
            logger.error(f"Error caching data: {e}")
            if self.session.is_active:
                self.session.rollback()
            raise
